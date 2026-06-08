#include "graphics/shadow_render_list.h"

#include "graphics/grinternal.h"
#include "graphics/matrix.h"
#include "graphics/shadows.h"
#include "model/model.h"
#include "render/3d.h"

extern float model_render_determine_depth(int obj_num, int model_num, const matrix* orient, const vec3d* pos, int detail_level_locked);
extern int model_render_determine_detail(float depth, int model_num, int detail_level_locked);

shadow_render_list::shadow_render_list()
{
}

shadow_render_list::~shadow_render_list()
{
}

void shadow_render_list::reset()
{
	_batches.clear();
}

void shadow_render_list::add_draw(const indexed_vertex_source* vert_src,
                                  vertex_buffer* buffer,
                                  size_t texi,
                                  const matrix4& model_matrix,
                                  const vec3d& scale,
                                  const clip_plane_info* clip)
{
	batch_key key;
	key.vert_src = vert_src;
	key.buffer = buffer;
	key.texi = texi;

	batch_entry entry;
	entry.model_matrix = model_matrix;
	entry.scale = scale;
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

void shadow_render_list::build_and_render(const matrix4* shadow_proj_matrices)
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
		auto& entries = kv.second;

		for (auto& entry : entries) {
			auto element = _dataBuffer.aligner().addTypedElement<graphics::shadow_uniform_data>();

			// Exact replica of convert_model_material lines 43-48
			matrix4 scaled_matrix = entry.model_matrix;
			scaled_matrix.a2d[0][0] *= entry.scale.xyz.x;
			scaled_matrix.a2d[0][1] *= entry.scale.xyz.x;
			scaled_matrix.a2d[0][2] *= entry.scale.xyz.x;
			scaled_matrix.a2d[1][0] *= entry.scale.xyz.y;
			scaled_matrix.a2d[1][1] *= entry.scale.xyz.y;
			scaled_matrix.a2d[1][2] *= entry.scale.xyz.y;
			scaled_matrix.a2d[2][0] *= entry.scale.xyz.z;
			scaled_matrix.a2d[2][1] *= entry.scale.xyz.z;
			scaled_matrix.a2d[2][2] *= entry.scale.xyz.z;
			element->modelMatrix = scaled_matrix;
			vm_matrix4_x_matrix4(&element->modelViewMatrix, &gr_view_matrix, &scaled_matrix);

			for (size_t i = 0; i < MAX_SHADOW_CASCADES; i++) {
				element->shadow_proj_matrix[i] = shadow_proj_matrices[i];
			}

			element->clip_equation = entry.clip_equation;
			element->use_clip_plane = entry.has_clip_plane ? 1 : 0;

			entry.uniform_buffer_offset = _dataBuffer.getCurrentAlignerOffset();
		}
	}

	_dataBuffer.submitData();

	for (auto& kv : _batches) {
		auto& entries = kv.second;

		for (size_t i = 0; i < entries.size(); i++) {
			auto* datap = &kv.first.buffer->tex_buf[kv.first.texi];
			if (datap->n_verts == 0) {
				continue;
			}

			gr_render_shadow_draw(
				_dataBuffer.bufferHandle(),
				entries[i].uniform_buffer_offset,
				sizeof(graphics::shadow_uniform_data),
				kv.first.buffer,
				const_cast<indexed_vertex_source*>(kv.first.vert_src),
				kv.first.texi);
		}
	}

	gr_alpha_mask_set(0, 1.0f);
}

bool shadow_render_list::batch_key::operator<(const batch_key& other) const
{
	if (vert_src != other.vert_src) return vert_src < other.vert_src;
	if (buffer != other.buffer) return buffer < other.buffer;
	return texi < other.texi;
}

void shadow_render_list::push_transform(const vec3d* pos, const matrix* orient)
{
	_transform_stack.push(pos, orient);
}

void shadow_render_list::pop_transform()
{
	_transform_stack.pop();
}

const matrix4& shadow_render_list::get_current_transform() const
{
	return _transform_stack.get_transform();
}

void shadow_render_list::add_model_draws(shadow_render_list* list,
                                         polymodel* pm,
                                         polymodel_instance* pmi,
                                         int obj_num,
                                         const vec3d* pos, const matrix* orient,
                                         const clip_plane_info* clip)
{
	float depth = model_render_determine_depth(obj_num, pm->id, orient, pos, -1);
	int detail_level = model_render_determine_detail(depth, pm->id, -1);
	int detail_root = pm->detail[detail_level];

	//list->_transform_stack.clear();
	list->push_transform(pos, orient);

	const vec3d scale_identity = SCALE_IDENTITY_VECTOR;

	if (pm->flags & PM_FLAG_AUTOCEN) {
		vec3d auto_back = pm->autocenter;
		vm_vec_scale(&auto_back, -1.0f);
		list->push_transform(&auto_back, NULL);
	}

	// Render the detail root submodel (hull)
	{
		auto& buffer = pm->submodel[detail_root].buffer;
		if (!buffer.tex_buf.empty()) {
			matrix4 world = list->get_current_transform();

			for (size_t j = 0; j < buffer.tex_buf.size(); j++) {
				if (buffer.tex_buf[j].n_verts == 0) {
					continue;
				}

				int tmap_num = buffer.tex_buf[j].texture;
				int base_tex = pm->maps[tmap_num].textures[TM_BASE_TYPE].GetTexture();

				if (base_tex < 0) {
					continue;
				}

				list->add_draw(&pm->vert_source, &buffer, j, world, scale_identity, clip);
			}
		}
	}

	// Walk children and render
	int i = pm->submodel[detail_root].first_child;

	while (i >= 0) {
		if (!pm->submodel[i].flags[Model::Submodel_flags::Is_thruster]) {
			render_submodel_children(list, pm, pmi, i, clip);
		}

		i = pm->submodel[i].next_sibling;
	}

	if (pm->flags & PM_FLAG_AUTOCEN) {
		list->pop_transform();
	}

	list->pop_transform();
}

void shadow_render_list::render_submodel_children(shadow_render_list* list,
                                                  polymodel* pm,
                                                  polymodel_instance* pmi,
                                                  int mn,
                                                  const clip_plane_info* clip)
{
	bsp_info* sm = &pm->submodel[mn];
	submodel_instance* smi = nullptr;

	if (pmi != nullptr) {
		smi = &pmi->submodel[mn];
		if (smi->blown_off) {
			return;
		}
	}

	matrix submodel_orient = vmd_identity_matrix;
	vec3d submodel_offset = sm->offset;

	const vec3d scale_identity = SCALE_IDENTITY_VECTOR;

	if (smi != nullptr) {
		submodel_orient = smi->canonical_orient;
		vm_vec_add2(&submodel_offset, &smi->canonical_offset);
	}

	list->push_transform(&submodel_offset, &submodel_orient);

	// Render this submodel's geometry
	{
		auto& buffer = sm->buffer;
		if (!buffer.tex_buf.empty()) {
			matrix4 world = list->get_current_transform();

			for (size_t j = 0; j < buffer.tex_buf.size(); j++) {
				if (buffer.tex_buf[j].n_verts == 0) {
					continue;
				}

				int tmap_num = buffer.tex_buf[j].texture;
				int base_tex = pm->maps[tmap_num].textures[TM_BASE_TYPE].GetTexture();

				if (base_tex < 0) {
					continue;
				}

				list->add_draw(&pm->vert_source, &buffer, j, world, scale_identity, clip);
			}
		}
	}

	// Recurse into children
	int i = sm->first_child;
	while (i >= 0) {
		if (!pm->submodel[i].flags[Model::Submodel_flags::Is_thruster]) {
			render_submodel_children(list, pm, pmi, i, clip);
		}

		i = pm->submodel[i].next_sibling;
	}

	list->pop_transform();
}
