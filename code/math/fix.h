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
	explicit constexpr fix(float in) : _fix(static_cast<std::int32_t>((in) * 65536.0f)) { }

	explicit constexpr operator int() const	{
		return static_cast<int>(_fix >> 16);
	}

	constexpr operator float() const {
		return static_cast<float>(_fix) / 65536.0f;
	}
	constexpr operator double() const {
		return static_cast<double>(_fix) / 65536.0;
	}

	constexpr std::int32_t get_raw() const {
		return _fix;
	}

	constexpr static fix set_raw(int in) {
		//This is somewhat ugly, but we definitely want the cast-to-fix constructor to correctly keep the int content
		fix f{};
		f._fix = in;
		return f;
	}

	constexpr fix& operator-=(const fix& other) {
		_fix -= other._fix;
		return *this;
	}

	constexpr fix& operator+=(const fix& other) {
		_fix += other._fix;
		return *this;
	}

	constexpr fix& operator*=(const int& other) {
		_fix *= other;
		return *this;
	}

	constexpr fix& operator/=(const int& other) {
		_fix /= other;
		return *this;
	}

	constexpr fix operator%=(const fix& r) {
		_fix %= r._fix;
		return *this;
	}

	/*
	 * fix fixmul(fix a, fix b)
{
	longlong tmp;
	tmp = (longlong)a * (longlong)b;
	return (fix)(tmp>>16);
}

	fix fixdiv(fix a, fix b)
	{
		return MulDiv(a,65536,b);
	}

	fix fixmuldiv(fix a, fix b,fix c)
	{
		return MulDiv(a,b,c);
	}

	 */

	constexpr bool operator<(const fix& other) const {
		return _fix < other._fix;
	}

	constexpr bool operator>(const fix& other) const {
		return _fix > other._fix;
	}

	constexpr bool operator<=(const fix& other) const {
		return !(*this > other);
	}

	constexpr bool operator>=(const fix& other) const {
		return !(*this < other);
	}

	constexpr bool operator==(const fix& other) const {
		return _fix == other._fix;
	}

	constexpr bool operator!=(const fix& other) const {
		return !(*this == other);
	}
};

constexpr fix operator-(fix l, const fix& r) {
	return l -= r;
}

constexpr fix operator+(fix l, const fix& r) {
	return l += r;
}

constexpr fix operator*(fix l, const int& r) {
	return l *= r;
}

constexpr fix operator*(const int& l, const fix& r) {
	return r * l;
}

constexpr fix operator/(fix l, const int& r) {
	return l /= r;
}

constexpr fix operator%(fix l, const fix& r) {
	return l %= r;
}

static constexpr fix F1_0(1);

#endif
