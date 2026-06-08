#pragma once

#include "globalincs/pstypes.h"
#include "graphics/util/uniform_structs.h"
#include "graphics/2d.h"
#include "graphics/util/UniformBuffer.h"
#include "matrix.h"

class shadow_render_list {
public:
	struct clip_plane_info {
		vec3d normal;
		vec3d position;
	};

	shadow_render_list();
	~shadow_render_list();

	void reset();

	size_t alloc_transform(int n_models);

	void add_draw(const indexed_vertex_source* vert_src,
	              vertex_buffer* buffer,
	              size_t texi,
	              size_t transform_base_offset,
	              const matrix4& model_matrix,
	              const clip_plane_info* clip);

	void submit_transforms();

	void build_and_render(const matrix4& light_view_matrix,
	                      const matrix4* shadow_proj_matrices);

	// Walk all submodels of a polymodel and add shadow draws
	static void add_model_draws(shadow_render_list* list,
	                            class polymodel* pm,
	                            size_t transform_base_offset,
	                            const matrix4& world_matrix,
	                            const clip_plane_info* clip);

private:
	struct batch_key {
		const indexed_vertex_source* vert_src;
		vertex_buffer* buffer;
		size_t texi;

		bool operator<(const batch_key& other) const;
	};

	struct batch_entry {
		size_t transform_base_offset;
		bool has_clip_plane;
		vec4 clip_equation;
		matrix4 model_matrix;
	};

	SCP_vector<matrix4> _transforms;
	SCP_map<batch_key, SCP_vector<batch_entry>> _batches;
	graphics::util::UniformBuffer _dataBuffer;
	size_t _current_transform_offset;
};
