/*
 * Copyright (C) Volition, Inc. 1999.  All rights reserved.
 *
 * All source code herein is the property of Volition, Inc. You may not sell 
 * or otherwise commercially exploit the source or things you created based on the 
 * source.
 *
*/ 



#ifndef _VECMAT_H
#define _VECMAT_H

#include "globalincs/pstypes.h"
#include "math/floating.h"
#include <limits>


#define vm_is_vec_nan(v) (fl_is_nan((v)->xyz.x) || fl_is_nan((v)->xyz.y) || fl_is_nan((v)->xyz.z))

//Macros/functions to fill in fields of structures
//VEC_NULL macros split into two functions in 2009 with commit 75a514b

// Null vector checks are performed on the following types of vectors:
// * orientation component vectors
// * positions
// * velocities
// * normals
// In each of these cases, FLT_EPSILON or 1.192092896e-07F is a reasonable threshold.

// macro to check if vector is close to zero or would be close to zero after squaring
// (uses FLT_EPSILON; original threshold was 1e-16 which can be tightened up a bit)
#define IS_VEC_NULL_SQ_SAFE(v) \
		(fl_near_zero((v)->xyz.x) && \
		fl_near_zero((v)->xyz.y) && \
		fl_near_zero((v)->xyz.z))

// macro to check if vector is close to zero
// (original threshold was 1e-36 which was too small)
#define IS_VEC_NULL(v) IS_VEC_NULL_SQ_SAFE(v)

// macro to check if moment-of-inertia vector is close to zero
// (uses the previous 1e-36 threshold since MOI values are really small)
#define IS_MOI_VEC_NULL(v) \
		(fl_near_zero((v)->xyz.x, (float) 1e-36) && \
		fl_near_zero((v)->xyz.y, (float) 1e-36) && \
		fl_near_zero((v)->xyz.z, (float) 1e-36))

// currently only used to check orientations
#define IS_MAT_NULL(v) (IS_VEC_NULL(&(v)->vec.fvec) && IS_VEC_NULL(&(v)->vec.uvec) && IS_VEC_NULL(&(v)->vec.rvec))

//macro to set a vector to zero.  we could do this with an in-line assembly
//macro, but it's probably better to let the compiler optimize it.
//Note: NO RETURN VALUE
#define vm_vec_zero(v) (v)->xyz.x=(v)->xyz.y=(v)->xyz.z=0.0f

//macro to set a vector to zero.  we could do this with an in-line assembly
//macro, but it's probably better to let the compiler optimize it.
//Note: NO RETURN VALUE
#define vm_mat_zero(m) (vm_vec_zero(&(m)->vec.rvec), vm_vec_zero(&(m)->vec.uvec), vm_vec_zero(&(m)->vec.fvec))

/*
//macro to set a matrix to the identity. Note: NO RETURN VALUE
#define vm_set_identity(m) do {m->rvec.x = m->uvec.y = m->fvec.z = (float)1.0;	\
										m->rvec.y = m->rvec.z = \
										m->uvec.x = m->uvec.z = \
										m->fvec.x = m->fvec.y = (float)0.0;} while (0)
*/
extern void vm_set_identity(matrix *m);

#define vm_vec_make(v,_x,_y,_z) ((v)->xyz.x=(_x), (v)->xyz.y=(_y), (v)->xyz.z=(_z))

extern angles vm_angles_new(float p, float b, float h);
extern vec3d vm_vec_new(float x, float y, float z);
extern vec4 vm_vec4_new(float x, float y, float z, float w);
extern matrix vm_matrix_new(float a0, float a1, float a2, float a3, float a4, float a5, float a6, float a7, float a8);
extern matrix vm_matrix_new(vec3d rvec, vec3d uvec, vec3d fvec);

//Global constants

extern vec3d vmd_zero_vector;
extern vec3d vmd_scale_identity_vector;
extern vec3d vmd_x_vector;
extern vec3d vmd_y_vector;
extern vec3d vmd_z_vector;
extern matrix vmd_zero_matrix;
extern matrix vmd_identity_matrix;
extern matrix4 vmd_zero_matrix4;
extern angles vmd_zero_angles;

//Here's a handy constant
#define ZERO_ANGLES { 0.0f, 0.0f, 0.0f }
#define ZERO_VECTOR { { { 0.0f, 0.0f, 0.0f } } }
#define SCALE_IDENTITY_VECTOR { { { 1.0f, 1.0f, 1.0f } } }
//#define IDENTITY_MATRIX {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}
// first set of inside braces is for union, second set is for inside union, then for a2d[3][3] (some compiler warning messages just suck)
//#define IDENTITY_MATRIX { { { {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} } } }
#define IDENTITY_MATRIX { { { { { { 1.0f, 0.0f, 0.0f } } }, { { { 0.0f, 1.0f, 0.0f } } }, { { { 0.0f, 0.0f, 1.0f } } } } } }
#define ZERO_MATRIX { { { ZERO_VECTOR, ZERO_VECTOR, ZERO_VECTOR } } }
#define ZERO_VECTOR4 { { { 0.0f, 0.0f, 0.0f, 0.0f } } }
#define ZERO_MATRIX4 { { { ZERO_VECTOR4, ZERO_VECTOR4, ZERO_VECTOR4, ZERO_VECTOR4 } } }

//fills in fields of an angle vector
#define vm_angvec_make(v,_p,_b,_h) (((v)->p=(_p), (v)->b=(_b), (v)->h=(_h)), (v))

//negate a vector
#define vm_vec_negate(v) do {(v)->xyz.x = - (v)->xyz.x; (v)->xyz.y = - (v)->xyz.y; (v)->xyz.z = - (v)->xyz.z;} while (false);

typedef struct plane {
	float	A, B, C, D;
} plane;

//Functions in library

//adds two vectors, fills in dest, returns ptr to dest
//ok for dest to equal either source, but should use vm_vec_add2() if so
//dest = src0 + src1
inline void vm_vec_add(vec3d *dest, const vec3d *src0, const vec3d *src1)
{
	dest->xyz.x = src0->xyz.x + src1->xyz.x;
	dest->xyz.y = src0->xyz.y + src1->xyz.y;
	dest->xyz.z = src0->xyz.z + src1->xyz.z;
}

//Component-wise multiplication of two vectors
inline void vm_vec_cmult(vec3d* dest, const vec3d* src0, const vec3d* src1) {
	dest->xyz.x = src0->xyz.x * src1->xyz.x;
	dest->xyz.y = src0->xyz.y * src1->xyz.y;
	dest->xyz.z = src0->xyz.z * src1->xyz.z;
}
inline void vm_vec_cmult2(vec3d* dest, const vec3d* src) {
	dest->xyz.x *= src->xyz.x;
	dest->xyz.y *= src->xyz.y;
	dest->xyz.z *= src->xyz.z;
}

//Component-wise division of two vectors
inline void vm_vec_cdiv(vec3d* dest, const vec3d* src0, const vec3d* src1) {
	dest->xyz.x = src0->xyz.x / src1->xyz.x;
	dest->xyz.y = src0->xyz.y / src1->xyz.y;
	dest->xyz.z = src0->xyz.z / src1->xyz.z;
}
inline void vm_vec_cdiv2(vec3d* dest, const vec3d* src) {
	dest->xyz.x /= src->xyz.x;
	dest->xyz.y /= src->xyz.y;
	dest->xyz.z /= src->xyz.z;
}

//subs two vectors, fills in dest, returns ptr to dest
//ok for dest to equal either source, but should use vm_vec_sub2() if so
//dest = src0 - src1
inline void vm_vec_sub(vec3d *dest, const vec3d *src0, const vec3d *src1)
{
	dest->xyz.x = src0->xyz.x - src1->xyz.x;
	dest->xyz.y = src0->xyz.y - src1->xyz.y;
	dest->xyz.z = src0->xyz.z - src1->xyz.z;
}


//adds one vector to another. returns ptr to dest
//dest can equal source
//dest += src
inline void vm_vec_add2(vec3d *dest, const vec3d *src)
{
	dest->xyz.x += src->xyz.x;
	dest->xyz.y += src->xyz.y;
	dest->xyz.z += src->xyz.z;
}

//subs one vector from another, returns ptr to dest
//dest can equal source
//dest -= src
inline void vm_vec_sub2(vec3d *dest, const vec3d *src)
{
	dest->xyz.x -= src->xyz.x;
	dest->xyz.y -= src->xyz.y;
	dest->xyz.z -= src->xyz.z;
}

//averages two vectors. returns ptr to dest
//dest can equal either source
//dest = (src0 + src1) * 0.5
inline vec3d *vm_vec_avg(vec3d *dest, const vec3d *src0, const vec3d *src1)
{
	dest->xyz.x = (src0->xyz.x + src1->xyz.x) * 0.5f;
	dest->xyz.y = (src0->xyz.y + src1->xyz.y) * 0.5f;
	dest->xyz.z = (src0->xyz.z + src1->xyz.z) * 0.5f;

	return dest;
}

//averages three vectors. returns ptr to dest
//dest can equal any source
//dest = (src0 + src1 + src2) *0.33
inline vec3d *vm_vec_avg3(vec3d *dest, const vec3d *src0, const vec3d *src1, const vec3d *src2)
{
	dest->xyz.x = (src0->xyz.x + src1->xyz.x + src2->xyz.x) * 0.333333333f;
	dest->xyz.y = (src0->xyz.y + src1->xyz.y + src2->xyz.y) * 0.333333333f;
	dest->xyz.z = (src0->xyz.z + src1->xyz.z + src2->xyz.z) * 0.333333333f;
	return dest;
}

//averages four vectors. returns ptr to dest
//dest can equal any source
//dest = (src0 + src1 + src2 + src3) * 0.25
inline vec3d *vm_vec_avg4(vec3d *dest, const vec3d *src0, const vec3d *src1, const vec3d *src2, const vec3d *src3)
{
	dest->xyz.x = (src0->xyz.x + src1->xyz.x + src2->xyz.x + src3->xyz.x) * 0.25f;
	dest->xyz.y = (src0->xyz.y + src1->xyz.y + src2->xyz.y + src3->xyz.y) * 0.25f;
	dest->xyz.z = (src0->xyz.z + src1->xyz.z + src2->xyz.z + src3->xyz.z) * 0.25f;
	return dest;
}


//scales a vector in place.
//dest *= s
inline void vm_vec_scale(vec3d *dest, float s)
{
	dest->xyz.x = dest->xyz.x * s;
	dest->xyz.y = dest->xyz.y * s;
	dest->xyz.z = dest->xyz.z * s;
}

//scales a 4-component vector in place.
// dest *= s
inline void vm_vec_scale(vec4 *dest, float s)
{
	dest->xyzw.x = dest->xyzw.x * s;
	dest->xyzw.y = dest->xyzw.y * s;
	dest->xyzw.z = dest->xyzw.z * s;
	dest->xyzw.w = dest->xyzw.w * s;
}

//scales and copies a vector.
// dest = src * s
inline void vm_vec_copy_scale(vec3d *dest, const vec3d *src, float s)
{
	dest->xyz.x = src->xyz.x*s;
	dest->xyz.y = src->xyz.y*s;
	dest->xyz.z = src->xyz.z*s;
}

//scales a vector, adds it to another, and stores in a 3rd vector
//dest = src1 + k * src2
inline void vm_vec_scale_add(vec3d *dest, const vec3d *src1, const vec3d *src2, float k)
{
	dest->xyz.x = src1->xyz.x + src2->xyz.x*k;
	dest->xyz.y = src1->xyz.y + src2->xyz.y*k;
	dest->xyz.z = src1->xyz.z + src2->xyz.z*k;
}

//scales a vector, subtracts it from another, and stores in a 3rd vector
//dest = src1 - (k * src2)
inline void vm_vec_scale_sub(vec3d *dest, const vec3d *src1, const vec3d *src2, float k)
{
	dest->xyz.x = src1->xyz.x - src2->xyz.x*k;
	dest->xyz.y = src1->xyz.y - src2->xyz.y*k;
	dest->xyz.z = src1->xyz.z - src2->xyz.z*k;
}

//scales a vector and adds it to another
//dest += k * src
inline void vm_vec_scale_add2(vec3d *dest, const vec3d *src, float k)
{
	dest->xyz.x += src->xyz.x*k;
	dest->xyz.y += src->xyz.y*k;
	dest->xyz.z += src->xyz.z*k;
}

//scales a vector and subtracts it from another
//dest -= k * src
inline void vm_vec_scale_sub2(vec3d *dest, const vec3d *src, float k)
{
	dest->xyz.x -= src->xyz.x*k;
	dest->xyz.y -= src->xyz.y*k;
	dest->xyz.z -= src->xyz.z*k;
}

//scales a vector in place, taking n/d for scale.
//dest *= n/d
inline void vm_vec_scale2(vec3d *dest, float n, float d)
{
	d = 1.0f/d;

	dest->xyz.x = dest->xyz.x* n * d;
	dest->xyz.y = dest->xyz.y* n * d;
	dest->xyz.z = dest->xyz.z* n * d;
}

// interpolate between two vectors
// dest = src0 + (k * (src1 - src0))
// Might be helpful to think of vec0 as the before, and vec1 as the after
inline void vm_vec_linear_interpolate(vec3d* dest, const vec3d* src0, const vec3d* src1, const float k)
{
	dest->xyz.x = ((src1->xyz.x - src0->xyz.x) * k) + src0->xyz.x;
	dest->xyz.y = ((src1->xyz.y - src0->xyz.y) * k) + src0->xyz.y;
	dest->xyz.z = ((src1->xyz.z - src0->xyz.z) * k) + src0->xyz.z;
}

//returns dot product of 2 vectors
inline float vm_vec_dot(const vec3d *v0, const vec3d *v1)
{
	return (v1->xyz.x*v0->xyz.x)+(v1->xyz.y*v0->xyz.y)+(v1->xyz.z*v0->xyz.z);
}


//returns dot product of <x,y,z> and vector
inline float vm_vec_dot3(float x, float y, float z, const vec3d *v)
{
	return (x*v->xyz.x)+(y*v->xyz.y)+(z*v->xyz.z);
}

//returns magnitude of a vector
inline float vm_vec_mag(const vec3d *v)
{
	float mag1;

	mag1 = (v->xyz.x * v->xyz.x) + (v->xyz.y * v->xyz.y) + (v->xyz.z * v->xyz.z);

	if (mag1 <= 0.0f) {
		return 0.0f;
	}

	return fl_sqrt(mag1);
}

//returns squared magnitude of a vector, useful if you want to compare distances
inline float vm_vec_mag_squared(const vec3d *v)
{
	return ((v->xyz.x * v->xyz.x) + (v->xyz.y * v->xyz.y) + (v->xyz.z * v->xyz.z));
}

//returns the square of the difference between v0 and v1 (the distance, squared)
//just like vm_vec_mag_squared, but the distance between two points instead.
inline float vm_vec_dist_squared(const vec3d *v0, const vec3d *v1)
{
	float dx, dy, dz;

	dx = v0->xyz.x - v1->xyz.x;
	dy = v0->xyz.y - v1->xyz.y;
	dz = v0->xyz.z - v1->xyz.z;
	return dx*dx + dy*dy + dz*dz;
}

//computes the distance between two points. (does sub and mag)
inline float vm_vec_dist(const vec3d *v0, const vec3d *v1)
{
	float t1;
	vec3d t;

	vm_vec_sub(&t,v0,v1);

	t1 = vm_vec_mag(&t);

	return t1;
}

inline bool vm_vec_is_normalized(const vec3d *v)
{
	// By the standards of FSO, it is sufficient to check that the magnitude is close to 1.
	return vm_vec_mag(v) > 0.999f && vm_vec_mag(v) < 1.001f;
}

//computes cross product of two vectors.
//Note: this magnitude of the resultant vector is the
//product of the magnitudes of the two source vectors.  This means it is
//quite easy for this routine to overflow and underflow.  Be careful that
//your inputs are ok.
//Dest cannot equal source
inline vec3d *vm_vec_cross(vec3d *dest, const vec3d *src0, const vec3d *src1)
{
	dest->xyz.x = (src0->xyz.y * src1->xyz.z) - (src0->xyz.z * src1->xyz.y);
	dest->xyz.y = (src0->xyz.z * src1->xyz.x) - (src0->xyz.x * src1->xyz.z);
	dest->xyz.z = (src0->xyz.x * src1->xyz.y) - (src0->xyz.y * src1->xyz.x);

	return dest;
}

//averages n vectors
vec3d *vm_vec_avg_n(vec3d *dest, int n, const vec3d src[]);

bool vm_vec_equal(const vec2d &self, const vec2d &other);

bool vm_vec_equal(const vec3d &self, const vec3d &other);

bool vm_vec_equal(const vec4 &self, const vec4 &other);

bool vm_matrix_equal(const matrix &self, const matrix &other);

bool vm_matrix_equal(const matrix4 &self, const matrix4 &other);

// finds the projection of source vector along a unit vector
// returns the magnitude of the component
float vm_vec_projection_parallel (vec3d *component, const vec3d *src, const vec3d *unit_vector);

// finds the projection of source vector onto a surface given by surface normal
void vm_vec_projection_onto_plane (vec3d *projection, const vec3d *src, const vec3d *normal);

// these are now deprecated because experimental testing on Discord has found
// that they are actually *slower* than their counterparts
#define vm_vec_mag_quick				vm_vec_mag
#define vm_vec_dist_quick				vm_vec_dist
#define vm_vec_copy_normalize_quick		vm_vec_copy_normalize
#define vm_vec_normalize_quick			vm_vec_normalize
#define vm_vec_normalized_dir_quick		vm_vec_normalized_dir
#define vm_vec_rand_vec_quick			vm_vec_rand_vec

//normalize a vector. returns mag of source vec
float vm_vec_copy_normalize(vec3d *dest, const vec3d *src);
float vm_vec_normalize(vec3d *v);

//	This version of vector normalize checks for the null vector before normalization.
//	If it is detected, it generates a Warning() and returns the vector 1, 0, 0.
float vm_vec_normalize_safe(vec3d *v);

//return the normalized direction vector between two points
//dest = normalized(end - start).  Returns mag of direction vector
// Returns mag of direction vector
//NOTE: the order of the parameters matches the vector subtraction
float vm_vec_normalized_dir(vec3d *dest,const vec3d *end, const vec3d *start);

float vm_vec_dot3(float x, float y, float z, vec3d *v);

/**
 * @brief Tests if the two vectors are parallel
 */
int vm_test_parallel(const vec3d *src0, const vec3d *src1);

//computes surface normal from three points. result is normalized
//returns ptr to dest
//dest CANNOT equal either source
vec3d *vm_vec_normal(vec3d *dest,const vec3d *p0, const vec3d *p1, const vec3d *p2);

//computes non-normalized surface normal from three points.
//returns ptr to dest
//dest CANNOT equal either source
vec3d *vm_vec_perp(vec3d *dest, const vec3d *p0, const vec3d *p1, const vec3d *p2);

//computes the delta angle between two vectors.
//vectors need not be normalized. if they are, call vm_vec_delta_ang_norm()
//the up vector (third parameter) can be NULL, in which case the absolute
//value of the angle in returned.  
//Otherwise, the delta ang will be positive if the v0 -> v1 direction from the
//point of view of uvec is clockwise, negative if counterclockwise.
//This vector should be orthogonal to v0 and v1
float vm_vec_delta_ang(const vec3d *v0, const vec3d *v1, const vec3d *uvec);

//computes the delta angle between two normalized vectors.
float vm_vec_delta_ang_norm(const vec3d *v0, const vec3d *v1,const vec3d *uvec);

//computes a matrix from a set of three angles.  returns ptr to matrix
matrix *vm_angles_2_matrix(matrix *m, const angles *a);

//	Computes a matrix from a single angle.
//	angle_index = 0,1,2 for p,b,h
matrix *vm_angle_2_matrix(matrix *m, float a, int angle_index);

//computes a matrix from a forward vector and an angle
matrix *vm_vec_ang_2_matrix(matrix *m, const vec3d *v, float a);

/**
 * @brief Generates a matrix from one or more vectors
 *
 * @param[out] matrix The generated matrix. Does not need to be an Identity matrix
 * @param[in] fvec Vector referencing the forward direction
 * @param[in] uvec Vector referencing the up direction (Optional)
 * @param[in] rvec Vector referencing the right-hand direction (Optional)
 *
 * @returns Pointer to the generated matrix
 *
 * @note If all three vectors are given, rvec is ignored.
 * @note If uvec was bogus (either being in the same direction of fvec or -fvec) then only fvec is used
 *
 * @sa vm_vector_2_matrix_norm
 */
matrix *vm_vector_2_matrix(matrix *m, const vec3d *fvec, const vec3d *uvec = nullptr, const vec3d *rvec = nullptr);


/**
 * @brief Generates a matrix from one or more normalized vectors
 *
 * @param[out] matrix The generated matrix. Does not need to be an Identity matrix
 * @param[in] fvec Normalized Vector referencing the forward direction
 * @param[in] uvec Normalized Vector referencing the up direction (Optional)
 * @param[in] rvec Normalized Vector referencing the right-hand direction (Optional)
 *
 * @returns Pointer to the generated matrix
 *
 * @note If all three vectors are given, rvec is ignored.
 * @note If uvec was bogus (either being in the same direction of fvec or -fvec) then only fvec is used
 *
 * @sa vm_vector_2_matrix
 */
matrix *vm_vector_2_matrix_norm(matrix *m, const vec3d *fvec, const vec3d *uvec = NULL, const vec3d *rvec = NULL);

//transpose a matrix in place. returns ptr to matrix
matrix *vm_transpose(matrix *m);

//copy and transpose a matrix. returns ptr to matrix
//dest CANNOT equal source. use vm_transpose() if this is the case
matrix *vm_copy_transpose(matrix *dest, const matrix *src);

//extract angles from a matrix
angles *vm_extract_angles_matrix(angles *a, const matrix *m);
angles *vm_extract_angles_matrix_alternate(angles *a, const matrix *m);

//extract heading and pitch from a vector, assuming bank==0
angles *vm_extract_angles_vector(angles *a, const vec3d *v);

//make sure matrix is orthogonal
void vm_orthogonalize_matrix(matrix *m_src);

// like vm_orthogonalize_matrix(), except that zero vectors can exist within the
// matrix without causing problems.  Valid vectors will be created where needed.
void vm_fix_matrix(matrix *m);

//Rotates the orient matrix by the angles in tangles and then
//makes sure that the matrix is orthogonal.
void vm_rotate_matrix_by_angles( matrix *orient, const angles *tangles );

//compute the distance from a point to a plane.  takes the normalized normal
//of the plane (ebx), a point on the plane (edi), and the point to check (esi).
//returns distance in eax
//distance is signed, so negative dist is on the back of the plane
float vm_dist_to_plane(const vec3d *checkp, const vec3d *norm, const vec3d *planep);

// Given mouse movement in dx, dy, returns a 3x3 rotation matrix in RotMat.
// Taken from Graphics Gems III, page 51, "The Rolling Ball"
// Example:
//if ( (Mouse.dx!=0) || (Mouse.dy!=0) ) {
//   vm_trackball( Mouse.dx, Mouse.dy, &MouseRotMat );
//   vm_matrix_x_matrix(&tempm,&LargeView.ev_matrix,&MouseRotMat);
//   LargeView.ev_matrix = tempm;
//}
void vm_trackball( int idx, int idy, matrix * RotMat );

//	Find the point on the line between p0 and p1 that is nearest to int_pnt.
//	Stuff result in nearest_point.
//	Return value indicated where on the line *nearest_point lies.  Between 0.0f and 1.0f means it's
//	in the line segment.  Positive means beyond *p1, negative means before *p0.  2.0f means it's
//	beyond *p1 by 2x.
float find_nearest_point_on_line(vec3d *nearest_point, const vec3d *p0, const vec3d *p1, const vec3d *int_pnt);

/**
 * @brief Find the intersection between two lines
 *
 * @param[out] s  If successful, s is the scalar of v0
 * @param[in]  p0 Reference point for line 1
 * @param[in]  p1 Reference point for line 2
 * @param[in]  v0 Direction vector for line 1 (must be normalized)
 * @param[in]  v1 Direction vector for line 2 (must be normalized)
 *
 * @returns  0 If successful, or
 * @returns -1 If colinear, or
 * @returns -2 If no intersection
 *
 * @note If you want the coords of the intersection, scale v0 by s, then add p0.
 */
int find_intersection(float *s, const vec3d* p0, const vec3d* p1, const vec3d* v0, const vec3d* v1);

/**
 * Finds the point on line 1 closest to line 2 when the lines are skew (non-intersecting in 3D space)
 *
 * @param[out] dest The closest point
 * @param[in]  p1 Reference point for line 1
 * @param[in]  d1 Direction vector for line 1 (must be normalized)
 * @param[in]  p2 Reference point for line 2
 * @param[in]  d2 Direction vector for line 2 (must be normalized)
 *
 * @note Algorithm from Wikipedia: https://en.wikipedia.org/wiki/Skew_lines#Formulas
 */
void find_point_on_line_nearest_skew_line(vec3d *dest, const vec3d *p1, const vec3d *d1, const vec3d *p2, const vec3d *d2);

// normalizes only if the vector's magnitude is above an optionally specified threshold, defaulting to 10 times machine epsilon
// returns whether or not it normalized
bool vm_maybe_normalize(vec3d* dst, const vec3d* src, float threshold = std::numeric_limits<float>::epsilon() * 10.f);

float vm_vec_dot_to_point(const vec3d *dir, const vec3d *p1, const vec3d *p2);

void compute_point_on_plane(vec3d *q, const plane *planep, const vec3d *p);

// ----------------------------------------------------------------------------
// computes the point on a plane closest to a given point (which may be on the plane)
// 
//		inputs:		new_point		=>		point on the plane [result]
//						point				=>		point to compute closest plane point
//						plane_normal	=>		plane normal
//						plane_point		=>		plane point
void vm_project_point_onto_plane(vec3d *new_point, const vec3d *point, const vec3d *plane_normal, const vec3d *plane_point);

//	Returns fairly random vector, normalized
void vm_vec_rand_vec(vec3d *rvec);

// Given an point "in" rotate it by "angle" around an
// arbritary line defined by a point on the line "line_point" 
// and the normalized line direction, "line_dir"
// Returns the rotated point in "out".
void vm_rot_point_around_line(vec3d *out, const vec3d *in, float angle, const vec3d *line_point, const vec3d *line_dir);

// Given two position vectors, return 0 if the same, else non-zero.
int vm_vec_cmp(const vec3d * a, const vec3d * b);

// Given two orientation matrices, return 0 if the same, else non-zero.
int vm_matrix_cmp(const matrix * a, const matrix * b);

// Moves angle 'h' towards 'desired_angle', taking the shortest
// route possible.   It will move a maximum of 'step_size' radians
// each call.   All angles in radians.
float vm_interp_angle( float *h, float desired_angle, float step_size, bool force_front = false );

// calculate and return the difference (ie. delta) between two angles
// using same method as with vm_interp_angle().
float vm_delta_from_interp_angle( float current_angle, float desired_angle );

// check a matrix for zero rows and columns
int vm_check_matrix_for_zeros(const matrix *m);

// see if two vectors are identical
int vm_vec_same(const vec3d *v1, const vec3d *v2);

// see if two matrices are identical
int vm_matrix_same(matrix *m1, matrix *m2);

// Interpolate from a start matrix toward a goal matrix, minimizing time between orientations.
// Moves at maximum rotational acceleration toward the goal when far and then max deceleration when close.
// Subject to constaints on rotational velocity and angular accleleration.
// Returns next_orientation valid at time delta_t.
// called "vm_matrix_interpolate" in retail 
void vm_angular_move_matrix(const matrix *goal_orient, const matrix *start_orient, const vec3d *rotvel_in, float delta_t,
		matrix *next_orient, vec3d *rotvel_out, const vec3d *rotvel_limit, const vec3d *acc_limit, bool no_directional_bias, bool force_no_overshoot = false);

// Interpolate from a start forward vec toward a goal forward vec, minimizing time between orientations.
// Moves at maximum rotational acceleration toward the goal when far and then max deceleration when close.
// Subject to constaints on rotational velocity and angular accleleration.
// Returns next forward vec valid at time delta_t.
// called "vm_forward_interpolate" in retail 
void vm_angular_move_forward_vec(const vec3d *goal_fvec, const matrix *orient, const vec3d *rotvel_in, float delta_t, float delta_bank,
		matrix *next_orient, vec3d *rotvel_out, const vec3d *vel_limit, const vec3d *acc_limit, bool no_directional_bias);

// Find the bounding sphere for a set of points (center and radius are output parameters)
void vm_find_bounding_sphere(const vec3d *pnts, int num_pnts, vec3d *center, float *radius);

// Translates from world coordinates to body coordinates
vec3d* vm_rotate_vec_to_body(vec3d *body_vec, const vec3d *world_vec, const matrix *orient);

// Translates from body coordinates to world coordiantes
vec3d* vm_rotate_vec_to_world(vec3d *world_vec, const vec3d *body_vec, const matrix *orient);

// estimate next orientation matrix as extrapolation of last and current
void vm_estimate_next_orientation(const matrix *last_orient, const matrix *current_orient, matrix *next_orient);

//	Return true if all elements of *vec are legal, that is, not a NAN.
bool is_valid_vec(const vec3d *vec);

//	Return true if all elements of *m are legal, that is, not a NAN.
bool is_valid_matrix(const matrix *m);

// Converts quaterions to a respective rotation matrix
void vm_quaternion_to_matrix(matrix* M, float a, float b, float c, float s);

// Finds the rotation matrix corresponding to a rotation of theta about axis u
void vm_quaternion_rotate(matrix *m, float theta, const vec3d *u);

// Takes a rotation matrix and returns the axis and angle needed to generate it
void vm_matrix_to_rot_axis_and_angle(const matrix *m, float *theta, vec3d *rot_axis);

// Given a rotation axis, calculates the angle that results in the rotation closest to the given matrix m. Returns the angle between the matrix orientation and the closest axis angle orientation
// If the axis is equal or very close to the orientation of the matrix, returns a distance of Pi/2 and an angle of 0
float vm_closest_angle_to_matrix(const matrix* mat, const vec3d* rot_axis, float* angle);

// interpolate between 2 vectors. t goes from 0.0 to 1.0
// out, v1 and v2 may all safely alias
void vm_vec_interp_constant(vec3d *out, const vec3d *v1, const vec3d *v2, float t);

// randomly perturb a vector around a given (normalized vector) or optional orientation matrix
void vm_vec_random_cone(vec3d *out, const vec3d *in, float max_angle, const matrix *orient = NULL);
void vm_vec_random_cone(vec3d *out, const vec3d *in, float min_angle, float max_angle, const matrix *orient = NULL);

// given a start vector, an orientation, and a radius, generate a point on the plane of the circle
// if on_edge is true, the point will be on the edge of the circle
// if bias_towards_center is true, the probability will be higher towards the center
void vm_vec_random_in_circle(vec3d *out, const vec3d *in, const matrix *orient, float radius, bool on_edge, bool bias_towards_center = false);


// compute a point on the unit sphere from cylindrical coordinate scale factors
// z_scale and phi_scale should be in [0.0, 1.0]
void vm_vec_unit_sphere_point(vec3d *out, float z_scale, float phi_scale);

// given a start vector and a radius, generate a point in a spherical volume
// if on_surface is true, the point will be on the surface of the sphere
// if bias_towards_center is true, the probability will be higher towards the center
void vm_vec_random_in_sphere(vec3d *out, const vec3d *in, float radius, bool on_surface, bool bias_towards_center = false);

// find the nearest point on the line to p. if dist is non-NULL, it is filled in
// returns 0 if the point is inside the line segment, -1 if "before" the line segment and 1 ir "after" the line segment
int vm_vec_dist_to_line(const vec3d *p, const vec3d *l0, const vec3d *l1, vec3d *nearest, float *dist);

// Goober5000
// Finds the distance squared to a line.  Same as above, except it uses vm_vec_dist_squared, which is faster;
// and it doesn't check whether the nearest point is on the line segment.
void vm_vec_dist_squared_to_line(const vec3d *p, const vec3d *l0, const vec3d *l1, vec3d *nearest, float *dist_squared);

//SUSHI: 2D vector "box" scaling
void vm_vec_boxscale(vec2d *vec, float scale);

void vm_matrix_add(matrix* dest, const matrix* src0, const matrix* src1);

void vm_matrix_sub(matrix* dest, const matrix* src0, const matrix* src1);

void vm_matrix_add2(matrix* dest, const matrix* src);

void vm_matrix_sub2(matrix* dest, const matrix* src);

bool vm_inverse_matrix(matrix* dest, const matrix* m);

bool vm_inverse_matrix4(matrix4* dest, const matrix4* m);

void vm_matrix4_set_orthographic(matrix4* out, vec3d *max, vec3d *min);

void vm_matrix4_set_inverse_transform(matrix4 *out, matrix *m, vec3d *v);

void vm_matrix4_set_identity(matrix4 *out);

void vm_matrix4_set_transform(matrix4 *out, matrix *m, vec3d *v);

void vm_matrix4_get_orientation(matrix *out, const matrix4 *m);

void vm_matrix4_get_offset(vec3d *out, const matrix4 *m);

void vm_vec_transform(vec4 *dest, const vec4 *src, const matrix4 *m);
void vm_vec_transform(vec3d *dest, const vec3d *src, const matrix4 *m, bool pos = true);

void vm_matrix4_x_matrix4(matrix4 *dest, const matrix4 *src0, const matrix4 *src1);

float vm_vec4_dot4(float x, float y, float z, float w, const vec4 *v);

/**
 * @brief Converts a 4 component vector to a 3 component vector by discarding the w component
 * @param vec The vector to convert
 * @return The converted 3 component vector
 */
vec3d vm_vec4_to_vec3(const vec4& vec);

/**
 * @brief Converts a 3 component vector to a 4 component vector with the specified w value
 * @param vec The first 3 components of the new vector
 * @param w The w component of the new vector. Defaults to 1.0f which is correct for position vectors.
 * @return The 4 component vector
 */
vec4 vm_vec3_to_ve4(const vec3d& vec, float w = 1.0f);

// calculates the best rvec to match another orient while maintaining a given fvec
void vm_match_bank(vec3d* out_rvec, const vec3d* goal_fvec, const matrix* match_orient);

// Interpolate between two matrices, using t as a percentage progress between them.
// Intended values for t are [0.0f, 1.0f], but values outside this range are allowed,
// as you could conceivably use these calculations to find a rotation that is outside 
// the usual 0-100%.
// derived by Asteroth from our AI code
void vm_interpolate_matrices(matrix* out_orient, const matrix* curr_orient, const matrix* goal_orient, float t);

// generates a well distributed quasi-random position in a -1 to 1 cube
// the caller must provide and increment the seed for each call for proper results
// if being used to fill a space, offset may be needed to properly 'glue together' generated
// volumes in a well distrubtedness-preserving way
vec3d vm_well_distributed_rand_vec(int seed, vec3d* offset = nullptr);

/** Compares two vec3ds */
inline bool operator==(const vec3d& left, const vec3d& right) { return vm_vec_same(&left, &right) != 0; }
inline bool operator!=(const vec3d& left, const vec3d& right) { return !(left == right); }

inline vec3d operator+(const vec3d& left, const vec3d& right)
{
	vec3d res;
	vm_vec_add(&res, &left, &right);
	return res;
}
inline vec3d& operator+=(vec3d& left, const vec3d& right)
{
	vm_vec_add2(&left, &right);
	return left;
}

inline vec3d operator-(const vec3d& left, const vec3d& right)
{
	vec3d res;
	vm_vec_sub(&res, &left, &right);
	return res;
}
inline vec3d& operator-=(vec3d& left, const vec3d& right)
{
	vm_vec_sub2(&left, &right);
	return left;
}

inline vec3d operator*(const vec3d& left, const vec3d& right)
{
	vec3d res;
	vm_vec_cmult(&res, &left, &right);
	return res;
}
inline vec3d& operator*=(vec3d& left, const vec3d& right)
{
	vm_vec_cmult2(&left, &right);
	return left;
}

inline vec3d operator/(const vec3d& left, const vec3d& right)
{
	vec3d res;
	vm_vec_cdiv(&res, &left, &right);
	return res;
}
inline vec3d& operator/=(vec3d& left, const vec3d& right)
{
	vm_vec_cdiv2(&left, &right);
	return left;
}

inline vec3d operator*(const vec3d& left, float right)
{
	vec3d out;
	vm_vec_copy_scale(&out, &left, right);
	return out;
}
inline vec3d operator*(float left, const vec3d& right)
{
	vec3d out;
	vm_vec_copy_scale(&out, &right, left);
	return out;
}
inline vec3d& operator*=(vec3d& left, float right)
{
	vm_vec_scale(&left, right);
	return left;
}

inline vec3d operator/(const vec3d& left, float right)
{
	vec3d out;
	vm_vec_copy_scale(&out, &left, 1.0f / right);
	return out;
}
inline vec3d& operator/=(vec3d& left, float right)
{
	vm_vec_scale(&left, 1.0f / right);
	return left;
}

inline matrix operator+(const matrix& left, const matrix& right)
{
	matrix res;
	vm_matrix_add(&res, &left, &right);
	return res;
}

inline matrix& operator+=(matrix& left, const matrix& right)
{
	vm_matrix_add2(&left, &right);
	return left;
}

inline matrix operator-(const matrix& left, const matrix& right)
{
	matrix res;
	vm_matrix_sub(&res, &left, &right);
	return res;
}

inline matrix& operator-=(matrix& left, const matrix& right)
{
	vm_matrix_sub2(&left, &right);
	return left;
}

inline angles& operator+=(angles& left, const angles& right)
{
	left.p += right.p;
	left.b += right.b;
	left.h += right.h;
	return left;
}

/**
 * @brief Implements matrix multiplication on 3D vectors
 * @param left The matrix
 * @param right The vector
 * @return The multiplied result
 */
inline vec3d operator*(const matrix& A, const vec3d& v)
{
	vec3d out;

	out.xyz.x = vm_vec_dot(&A.vec.rvec, &v);
	out.xyz.y = vm_vec_dot(&A.vec.uvec, &v);
	out.xyz.z = vm_vec_dot(&A.vec.fvec, &v);

	return out;
}

/**
 * @brief Implements matrix multiplication on 3x3 matrices
 * @param left The matrix
 * @param right The matrix
 * @return The multiplied result
 */
inline matrix operator*(const matrix& A, const matrix& B)
{
	matrix BT, out;

	// we transpose B here for concision and also potential vectorisation opportunities
	vm_copy_transpose(&BT, &B);

	out.vec.rvec = BT * A.vec.rvec;
	out.vec.uvec = BT * A.vec.uvec;
	out.vec.fvec = BT * A.vec.fvec;

	return out;
}

// rotates a vector through a matrix, writes to *dest and returns the pointer
// if m is a rotation matrix it will preserve the length of *src, so normalised vectors will remain normalised
inline vec3d *vm_vec_rotate(vec3d *dest, const vec3d *src, const matrix *m)
{
	*dest = (*m) * (*src);

	return dest;
}

//rotates a vector through the transpose of the given matrix.
//returns ptr to dest vector
// This is a faster replacement for this common code sequence:
//    vm_copy_transpose(&tempm,src_matrix);
//    vm_vec_rotate(dst_vec,src_vect,&tempm);
// Replace with:
//    vm_vec_unrotate(dst_vec,src_vect, src_matrix)
//
// THIS DOES NOT ACTUALLY TRANSPOSE THE SOURCE MATRIX!!! So if
// you need it transposed later on, you should use the
// vm_vec_transpose() / vm_vec_rotate() technique.
// like vm_vec_rotate, but uses the transpose matrix instead. for rotations, this is an inverse.
inline vec3d *vm_vec_unrotate(vec3d *dest, const vec3d *src, const matrix *m)
{
	matrix mt;

	vm_copy_transpose(&mt, m);
	*dest = mt * (*src);

	return dest;
}

// Old matrix multiplication routine. Note that the order of multiplication is inverted
// compared to the mathematical standard: formally, this calculates src1 * src0
inline matrix *vm_matrix_x_matrix(matrix *dest, const matrix *src0, const matrix *src1)
{
	*dest = (*src1) * (*src0);

	return dest;
}

std::ostream& operator<<(std::ostream& os, const vec3d& vec);

// Given a direction and a 'stretch amount', computes a matrix which can be used to
// 'rotate' positional vectors as to stretch them in that direction by that amount
// Positions in the opposite direction of the stretch_dir are stretched in the opposite direction
// and position orthogonal to the stretch_dir are not moved at all
// Essentially turns spheres into ellipsoids
matrix vm_stretch_matrix(const vec3d* stretch_dir, float stretch);

#endif


