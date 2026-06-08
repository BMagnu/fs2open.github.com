#pragma once

#include "globalincs/pstypes.h"
#include "graphics/util/uniform_structs.h"
#include "graphics/2d.h"
#include "graphics/util/UniformBuffer.h"
#include "matrix.h"

class polymodel;
class polymodel_instance;

class shadow_render_list {
public:
	struct clip_plane_info {
		vec3d normal;
		vec3d position;
	};

	shadow_render_list();
	~shadow_render_list();

	void reset();

	void add_draw(const indexed_vertex_source* vert_src,
	              vertex_buffer* buffer,
	              size_t texi,
	              const matrix4& model_matrix,
	              const vec3d& scale,
	              const clip_plane_info* clip);

	void push_transform(const vec3d* pos, const matrix* orient);
	void pop_transform();
	const matrix4& get_current_transform() const;

	void build_and_render(const matrix4* shadow_proj_matrices);

	// Walk all submodels of a polymodel and add shadow draws, using the transform_stack
	// for correct per-submodel world transforms.
	static void add_model_draws(shadow_render_list* list,
	                            polymodel* pm,
	                            polymodel_instance* pmi,
	                            int obj_num,
	                            const vec3d* pos, const matrix* orient,
	                            const clip_plane_info* clip);

private:
	struct batch_key {
		const indexed_vertex_source* vert_src;
		vertex_buffer* buffer;
		size_t texi;

		bool operator<(const batch_key& other) const;
	};

	struct batch_entry {
		size_t uniform_buffer_offset;
		bool has_clip_plane;
		vec4 clip_equation;
		matrix4 model_matrix;
		vec3d scale;
	};

	static void render_submodel_children(shadow_render_list* list,
	                                     polymodel* pm,
	                                     polymodel_instance* pmi,
	                                     int mn,
	                                     const clip_plane_info* clip);

	SCP_map<batch_key, SCP_vector<batch_entry>> _batches;
	graphics::util::UniformBuffer _dataBuffer;
	transform_stack _transform_stack;
};
