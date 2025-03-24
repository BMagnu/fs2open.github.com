/*
 * Copyright (C) Volition, Inc. 1999.  All rights reserved.
 *
 * All source code herein is the property of Volition, Inc. You may not sell 
 * or otherwise commercially exploit the source or things you created based on the 
 * source.
 *
*/

#ifndef _FIX_H
#define _FIX_H

class fix {
	std::int32_t _fix;

  public:
	constexpr fix() : _fix(0) {}
	explicit constexpr fix(int in) : _fix(in << 16) { }

	explicit constexpr operator int() const {
		return static_cast<int>(_fix >> 16);
	}
	constexpr operator float() const {
		return static_cast<float>(_fix) / 65536.0f;
	}
	constexpr operator double() const {
		return static_cast<double>(_fix) / 65536.0;
	}

	constexpr fix& operator-=(const fix& other) {
		_fix -= other._fix;
		return *this;
	}

	constexpr fix& operator+=(const fix& other) {
		_fix += other._fix;
		return *this;
	}
};

constexpr fix operator-(fix l, const fix& r) {
	return l -= r;
}

constexpr fix operator+(fix l, const fix& r) {
	return l += r;
}

static constexpr fix F1_0(1);

#endif
