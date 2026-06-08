#include "graphics/shadow_render_list.h"

#include "graphics/grinternal.h"
#include "graphics/matrix.h"
#include "graphics/shadows.h"
#include "model/model.h"

shadow_render_list::shadow_render_list()
	: _current_transform_offset(0)
{
	vm_matrix4_set_identity(&_identity_matrix);
}

shadow_render_list::~shadow_render_list()
{
}

void shadow_render_list::reset()
{
	_transforms.clear();
	_batches.clear();
	_current_transform_offset = 0;
}

size_t shadow_render_list::alloc_transform(const matrix4& world_matrix, int n_models)
{
	size_t offset = _current_transform_offset;
	_current_transform_offset += n_models;

	_transforms.resize(_current_transform_offset);

	for (int i = 0; i < n_models; i++) {
		_transforms[offset + i] = world_matrix;
	}

	return offset * sizeof(matrix4);
}

void shadow_render_list::add_draw(const indexed_vertex_source* vert_src,
                                  vertex_buffer* buffer,
                                  size_t texi,
                                  size_t transform_base_offset,
                                  const clip_plane_info* clip)
{
	batch_key key;
	key.vert_src = vert_src;
	key.buffer = buffer;
	key.texi = texi;

	batch_entry entry;
	entry.transform_base_offset = transform_base_offset;
	entry.has_clip_plane = (clip != nullptr);
	if (clip != nullptr) {
		entry.clip_equation.xyzw.x = clip->normal.xyz.x;
		entry.clip_equation.xyzw.y = clip->normal.xyz.y;
		entry.clip_equation.xyzw.z = clip->normal.xyz.z;
		entry.clip_equation.xyzw.w = -vm_vec_dot(&clip->normal, &clip->position);
	} else {
		entry.clip_equation.xyzw.x = 0.0f;
		entry.clip_equation.xyzw.y = 0.0f;
		entry.clip_equation.xyzw.z = 0.0f;
		entry.clip_equation.xyzw.w = 0.0f;
	}

	_batches[key].push_back(entry);
}

void shadow_render_list::submit_transforms()
{
	if (_transforms.empty()) {
		return;
	}

	gr_update_transform_buffer(_transforms.data(), _transforms.size() * sizeof(matrix4));
}

void shadow_render_list::build_and_render(const matrix4& light_view_matrix,
                                          const matrix4* shadow_proj_matrices)
{
	if (_batches.empty()) {
		return;
	}

	size_t total_entries = 0;
	for (const auto& kv : _batches) {
		total_entries += kv.second.size();
	}

	_dataBuffer = gr_get_uniform_buffer(uniform_block_type::ShadowMapData, total_entries);

	for (auto& kv : _batches) {
		const auto& entries = kv.second;

		for (const auto& entry : entries) {
			auto element = _dataBuffer.aligner().addTypedElement<graphics::shadow_uniform_data>();

			element->modelViewMatrix = light_view_matrix;
			element->modelMatrix = _identity_matrix;

			for (size_t i = 0; i < MAX_SHADOW_CASCADES; i++) {
				element->shadow_proj_matrix[i] = shadow_proj_matrices[i];
			}

			element->clip_equation = entry.clip_equation;
			element->use_clip_plane = entry.has_clip_plane ? 1 : 0;
			element->buffer_matrix_offset = static_cast<int>(entry.transform_base_offset / sizeof(matrix4));
		}
	}

	_dataBuffer.submitData();

	size_t entry_index = 0;
	for (auto& kv : _batches) {
		const auto& entries = kv.second;

		for (size_t i = 0; i < entries.size(); i++) {
			auto* datap = &kv.first.buffer->tex_buf[kv.first.texi];
			if (datap->n_verts == 0) {
				entry_index++;
				continue;
			}

			gr_render_shadow_draw(
				_dataBuffer.bufferHandle(),
				entry_index * sizeof(graphics::shadow_uniform_data),
				sizeof(graphics::shadow_uniform_data),
				kv.first.buffer,
				const_cast<indexed_vertex_source*>(kv.first.vert_src),
				kv.first.texi);

			entry_index++;
		}
	}
}

bool shadow_render_list::batch_key::operator<(const batch_key& other) const
{
	if (vert_src != other.vert_src) return vert_src < other.vert_src;
	if (buffer != other.buffer) return buffer < other.buffer;
	return texi < other.texi;
}

void shadow_render_list::add_model_draws(shadow_render_list* list,
                                         class polymodel* pm,
                                         size_t transform_base_offset,
                                         const clip_plane_info* clip)
{
	for (int i = 0; i < pm->n_models; i++) {
		if (i > 0 && pm->submodel[i].flags[Model::Submodel_flags::Is_thruster]) {
			continue;
		}

		auto& buffer = pm->submodel[i].buffer;
		if (buffer.tex_buf.empty()) {
			continue;
		}

		for (size_t j = 0; j < buffer.tex_buf.size(); j++) {
			if (buffer.tex_buf[j].n_verts == 0) {
				continue;
			}

			int tmap_num = buffer.tex_buf[j].texture;
			int base_tex = pm->maps[tmap_num].textures[TM_BASE_TYPE].GetTexture();

			if (base_tex < 0) {
				continue;
			}

			list->add_draw(&pm->vert_source, &buffer, j, transform_base_offset, clip);
		}
	}
}
