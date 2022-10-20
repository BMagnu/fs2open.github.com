/*
 * Copyright (C) Volition, Inc. 1999.  All rights reserved.
 *
 * All source code herein is the property of Volition, Inc. You may not sell 
 * or otherwise commercially exploit the source or things you created based on the 
 * source.
 *
*/ 



#ifndef _SNAZZYUI_H
#define _SNAZZYUI_H

#define MAX_CHAR		150
#define ESC_PRESSED	-2

#include "globalincs/pstypes.h"
#include "gamesnd/gamesnd.h"
#include "bmpman/bmpman.h"
#include "cfile/cfile.h"
#include "graphics/2d.h"

typedef struct menu_region {
	int 	mask;					// mask color for the region
	int	key;					// shortcut key for the region
	char	text[MAX_CHAR];	// The text associated with this item.
	interface_snd_id	click_sound;		// Id of sound to play when mask area clicked on
} MENU_REGION;

// These are the actions thare are returned in the action parameter.  
#define SNAZZY_OVER			1	// mouse is over a region
#define SNAZZY_CLICKED		2	// mouse button has gone from down to up over a region

int snazzy_menu_do(ubyte *data, int mask_w, int mask_h, int num_regions, MENU_REGION *regions, int *action, int poll_key = 1, int *key = NULL);
void read_menu_tbl(const char *menu_name, char *bkg_filename, char *mask_filename, MENU_REGION *regions, int* num_regions, int play_sound=1);
void snazzy_menu_add_region(MENU_REGION *region, const char* text, int mask, int key, interface_snd_id click_sound = interface_snd_id());

void snazzy_menu_init();		// Call the first time a snazzy menu is inited
void snazzy_menu_close();
void snazzy_flush();

// Bitmap availability validation code

template<typename T>
struct snazzyui_bitmap_filename_source {
	const char* T::* filename;
	inline const char* getFilename(const T& source) const { return source.*filename; }
};

template<>
struct snazzyui_bitmap_filename_source<const char*> {
	inline const char* getFilename(const char* source) const { return source; }
};

inline SCP_vector<SCP_string> snazzyui_bitmap_filename_frame_augmenter(const char* filename, size_t frames, size_t start) {
	if (frames == 0)
		return SCP_vector<SCP_string>{filename};

	SCP_vector<SCP_string> result;
	for (size_t i = start; i < frames; i++) {
		SCP_string zeroes{ "0000" };
		SCP_string number = std::to_string(i);
		zeroes.replace(4 - number.size(), number.size(), number);
		result.emplace_back(filename + zeroes);
	}

	return result;
}

template<typename T, size_t n>
struct snazzyui_bitmap_source {
	const T(&source)[GR_NUM_RESOLUTIONS][n];
	snazzyui_bitmap_filename_source<T> filename_getter;
	size_t frames, start;

	inline bool validate() const {
		for (const T(&obj)[n] : source) {
			for (const T& res : obj) {
				for (const SCP_string& fn : snazzyui_bitmap_filename_frame_augmenter(filename_getter.getFilename(res), frames, start)) {
					if (!cf_find_file_location_ext(fn.c_str(), BM_NUM_TYPES, bm_ext_list, CF_TYPE_ANY).found)
						return false;
				}
			}
		}
		return true;
	}
};

template<typename T>
struct snazzyui_bitmap_source<T, 0> {
	const T(&source)[GR_NUM_RESOLUTIONS];
	snazzyui_bitmap_filename_source<T> filename_getter;
	size_t frames, start;

	inline bool validate() const {
		for (const T& res : source) {
			for (const SCP_string& fn : snazzyui_bitmap_filename_frame_augmenter(filename_getter.getFilename(res), frames, start)) {
				if (!cf_find_file_location_ext(fn.c_str(), BM_NUM_TYPES, bm_ext_list, CF_TYPE_ANY).found)
					return false;
			}
		}
		return true;
	}
};

//Helper functions for automatic type deduction. No actual logic is performed here
template<typename T, size_t n> snazzyui_bitmap_source<T, n> inline make_snazzyui_bitmap_source(const T(&source)[GR_NUM_RESOLUTIONS][n], const char* T::* filename, size_t frames = 0, size_t start = 1) { return snazzyui_bitmap_source<T, n>{ source, snazzyui_bitmap_filename_source<T>{ filename }, frames, start }; }
template<typename T> snazzyui_bitmap_source<T, 0> inline make_snazzyui_bitmap_source(const T(&source)[GR_NUM_RESOLUTIONS], const char* T::* filename, size_t frames = 0, size_t start = 1) { return snazzyui_bitmap_source<T, 0>{ source, snazzyui_bitmap_filename_source<T>{ filename }, frames, start }; }
template<size_t n> snazzyui_bitmap_source<const char*, n> inline make_snazzyui_bitmap_source(const char* (&source)[GR_NUM_RESOLUTIONS][n], size_t frames = 0, size_t start = 1) { return snazzyui_bitmap_source<const char*, n>{ source, snazzyui_bitmap_filename_source<const char*>{} , frames, start }; }
snazzyui_bitmap_source<const char*, 0> inline make_snazzyui_bitmap_source(const char* (&source)[GR_NUM_RESOLUTIONS], size_t frames = 0, size_t start = 1) { return snazzyui_bitmap_source<const char*, 0>{ source, snazzyui_bitmap_filename_source<const char*>{}, frames, start }; }

template<typename T1, typename... T>
inline bool snazzy_validate_bitmaps_helper(const T1& arg, const T&... args) {
	if (!arg.validate())
		return false;

	return snazzy_validate_bitmaps_helper(args...);
}

template<typename T1>
inline bool snazzy_validate_bitmaps_helper(const T1& arg) {
	return arg.validate();
}

template<typename... T>
inline bool snazzy_validate_bitmaps(bool& hasValidBitmaps, const T&... args) {
	return (hasValidBitmaps = snazzy_validate_bitmaps_helper(args...));
}


#endif
