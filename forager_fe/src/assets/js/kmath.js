/*
 * kmath.js 
 *
 * This file includes useful datatypes and functions for basic math operations on
 * points, boxes, etc.
 */


export class Point2D {
	constructor(x, y) {
		this.x = x;
		this.y = y;
	}
}

export class BBox2D {
	constructor(startx, starty, width, height) {
		this.bmin = new Point2D(startx, starty);
		this.bmax = new Point2D(this.bmin.x + width, this.bmin.y + height);
	}

	// return width of the box
	get width() {
		if (this.is_empty())
			return 0.0;
		else
			return this.bmax.x - this.bmin.x;
	}

	// return height of the box
	get height() {
		if (this.is_empty())
			return 0.0;
		else
			return this.bmax.y - this.bmin.y;
	}

	// return area of bounding box
	get area() {
		if (this.is_empty()) 
			return 0.0;
		else
			return (this.bmax.x - this.bmin.x) * (this.bmax.y - this.bmin.y);
	}

	// returns true if the bounding box contains zero area
	// 
	// FIXME(kayvonf): currently defining the bbox as *containing* both the top/left and
	// the bottom/right edge. This is unconventional, revist this later.  I'd prefer to
	// include the top/left (min) edges, and not include the bottom/right edges. 
	// Note make this fix may impact the ability to sucessfully select points.  (Which is
	// why it was postponed.)
	is_empty() {
		return (this.bmin.x > this.bmax.x || this.bmin.y > this.bmax.y);
	}

	// returns true if the point (x,y) is inside the bounding box.
	// 
	// FIXME(kayvonf): currently defining the bbox as *containing* the
	// both the top/left and bottom/right edges. This is unconventional,
	// revist this later.
	inside(pt) {
		return (pt.x >= this.bmin.x && pt.x <= this.bmax.x &&
   			    pt.y >= this.bmin.y && pt.y <= this.bmax.y);
	}

	// Returns a new bbox that is a scaled version of this one. scaling given by (sx,sy)
	scale(sx, sy) {
		var minx = this.bmin.x * sx;
		var miny = this.bmin.y * sy;
		var maxx = this.bmax.x * sx;
		var maxy = this.bmax.y * sy;
		return new BBox2D(minx, miny, maxx-minx, maxy-miny)
	}

	// Returns a new bbox that is this bbox intersected with the provided bbox
	intersect(clip_bbox) {
		var minx = Math.max(this.bmin.x, clip_bbox.bmin.x);
		var miny = Math.max(this.bmin.y, clip_bbox.bmin.y);
		var maxx = Math.min(this.bmax.x, clip_bbox.bmax.x);
		var maxy = Math.min(this.bmax.y, clip_bbox.bmax.y);
		return new BBox2D(minx, miny, maxx-minx, maxy-miny);
	}

	to_string() {
		return "min=(" + this.bmin.x + ", " + this.bmin.y + "), max=(" + this.bmax.x + ", " + this.bmax.y + "), w=" + this.width + ", h=" + this.height + ", e=" + this.is_empty();
	}	

	// Converts an array of four extreme points to a bbox
	// NOTE(kayvonf): This code assumes that the extreme points are provided
	// in a canonical (clockwise order beginning with the leftmost point)
	static extreme_points_to_bbox(pts) {

		// pts[0] = left extreme
		// pts[1] = top extreme
		// pts[2] = right extreme
		// pts[3] = bottom extreme

		var startx = pts[0].x;
		var starty = pts[1].y;
		var endx = pts[2].x;
		var endy = pts[3].y;

		var b = new BBox2D(startx, starty, endx - startx, endy - starty);
		return b;
	}

	// Converts an array of two points to a bbox.  The code makes no
	// assumptions about which point in the array is which corner of
	// the bbox (it figures it out) 
	static two_points_to_bbox(pts) {

		var startx = Math.min(pts[0].x, pts[1].x);
		var starty = Math.min(pts[0].y, pts[1].y);
		var endx = Math.max(pts[0].x, pts[1].x);
		var endy = Math.max(pts[0].y, pts[1].y);

		var b = new BBox2D(startx, starty, endx - startx, endy - starty);
		return b;
	}

	// Returns true if this is a valid set of extreme points, false otherwise.
	//
	// To be valid, the first point should be the leftmost point, the second point
	// should be the uppermost one, the third point should be the rightmost,
	// and the fourth point should be bottommost. 
	static validate_extreme_points(pts) {

		if (pts[0].x > pts[1].x ||
			pts[0].x > pts[2].x ||
			pts[0].x > pts[3].x)
			return false;
		if (pts[1].y > pts[2].y ||
			pts[1].y > pts[3].y)
			return false;
		if (pts[2].x < pts[1].x ||
			pts[2].x < pts[3].x)
			return false;
		if (pts[3].y < pts[1].y ||
			pts[3].y < pts[2].y)
			return false;

		return true;
	}
}


export function clamp(x, min_value, max_value) {
	return Math.min(max_value, Math.max(min_value, x));
}
