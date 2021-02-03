// this file depends on:
//   -- kmath.js

/*
	This file contains the implementation of KLabel.

	TODO list:
	* improve selection mechanism for boxes/points
	* detect invalid extreme points as early as possible (before end of box)

	LICENSES OF THINGS I USED:
	* This is the "click1" sound:
	  https://www.zapsplat.com/music/metal-impact-small-plate-disc-hit-vibrate-and-resonate-2/
*/

import { BBox2D, Point2D, clamp } from "./kmath.js";

export class Annotation {

	static get INVALID_CATEGORY() { return -1; }

	static get ANNOTATION_MODE_PER_FRAME_CATEGORY() { return 0; }
	static get ANNOTATION_MODE_POINT() { return 1; }
	static get ANNOTATION_MODE_TWO_POINTS_BBOX() { return 2; }
	static get ANNOTATION_MODE_EXTREME_POINTS_BBOX() { return 3; }

	constructor(type) {
		this.type = type;
	}
}

export class PerFrameAnnotation extends Annotation {
	constructor(value) {
		super(Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY);
		this.value = value;
	}

	static parse(obj) {
		return Object.setPrototypeOf(obj, PerFrameAnnotation.prototype);
	}
}

export class PointAnnotation extends Annotation {
	constructor(pt) {
		super(Annotation.ANNOTATION_MODE_POINT);
		this.pt = pt;
	}

	static parse(obj) {
		obj.pt = Object.setPrototypeOf(obj.pt, Point2D.prototype);
		return Object.setPrototypeOf(obj, PointAnnotation.prototype);
	}
}

export class TwoPointBoxAnnotation extends Annotation {
	constructor(corner_pts) {
		super(Annotation.ANNOTATION_MODE_TWO_POINTS_BBOX);
		this.bbox = BBox2D.two_points_to_bbox(corner_pts);
	}

	static parse(obj) {
		obj.bbox = Object.setPrototypeOf(obj.bbox, BBox2D.prototype);
		return Object.setPrototypeOf(obj, TwoPointBoxAnnotation.prototype);
	}
}

export class ExtremeBoxAnnnotation extends Annotation {
	constructor(extreme_points) {
		super(Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX);
		this.bbox = BBox2D.extreme_points_to_bbox(extreme_points);
		this.extreme_points = extreme_points;
	}

	static parse(obj) {
		obj.bbox = Object.setPrototypeOf(obj.bbox, BBox2D.prototype);
		obj.extreme_points = obj.extreme_points.map(
			point => Object.setPrototypeOf(point, Point2D.prototype)
		)
		return Object.setPrototypeOf(obj, ExtremeBoxAnnnotation.prototype);
	}
}

export class ImageData {
	constructor() {
		this.source_url = "";
		this.annotations = [];
		this.labeling_time = 0.0;
	}
}

class Frame {
	constructor(image_data) {
		this.data = image_data;
		this.image_load_started = false;
		this.image_load_complete = false;
		this.source_image = new Image();
	}
}

export class ImageLabeler {

	constructor() {

		this.main_canvas_el = null;

		this.cursorx = Number.MIN_SAFE_INTEGER;
		this.cursory = Number.MIN_SAFE_INTEGER;

		// timing labeler behavior
		this.current_frame_start_time = null;

		// event callbacks
		this.frame_changed_callback = null;
		this.annotation_changed_callback = null;
		this.annotation_added_callback = null;
		this.annotation_deleted_callback = null;

		// state for UI related to zooming
		this.zoom_key_down = false;
		this.zoom_corner_points = [];

		// this structure holds the annotation data
		this.current_frame_index = 0;
		this.current_indices = [0];
		this.frames = [];

		// Hold current auto labeler id
		this.autoStepId = null;

		// annotation state
		this.annotation_mode = Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX;
		this.category_to_name = [];
		this.category_to_color = [];
		this.in_progress_points = [];

		// audio
		this.audio_click_sound = null;
		this.audio_box_done_sound = null;

		// colors
		this.color_background = '#202020';
		this.color_cursor_lines = 'rgba(0, 255, 255, 0.5)';
		this.color_in_progress_box_outline = 'rgba(255, 255, 255, 0.75)';
		this.color_box_outline = 'rgba(255,200,0,0.75)';
		this.color_selected_box_outline = 'rgba(255, 200, 100, 1.0)';
		this.color_selected_box_fill = 'rgba(255, 200, 150, 0.2)';
		this.color_extreme_point_fill = '#ffff00';
		this.color_point_fill = '#ffff00';
		this.color_selected_point_fill = '#ff0000';
		this.color_per_frame_annotation_outline = 'rgba(255, 50, 50, 0.5)';
		this.color_zoom_box_outline = 'rgba(255, 0, 0, 1.0)';
		this.color_zoom_box_fill = 'rgba(255, 0, 0, 0.3)';
		this.color_category_text_fill = '#000000';

		this.category_text_font = "16px Arial";

		// display settings
		this.visible_image_region = new BBox2D(0.0, 0.0, 1.0, 1.0);
		this.letterbox_image = true;  // if false, stretch image to fill canvas
		this.play_audio = false;
		this.show_crosshairs = true;
		this.show_extreme_points = true;
		this.retain_zoom = true;
		this.extreme_point_radius = 3;

	}

	update_labeling_time() {
	  this.frames[this.get_current_frame_num()].data.labeling_time += (
        performance.now() - this.current_frame_start_time);
		this.current_frame_start_time = performance.now();
	}

	// Clamp the cursor to the image dimensions so that clicks,
	// and (resulting bounding boxes) are always within the image
	set_canvas_cursor_position(x,y) {
		this.cursorx = clamp(x, 0, this.main_canvas_el.width);
		this.cursory = clamp(y, 0, this.main_canvas_el.height);	
	}

	get_current_frame() {
		this.update_labeling_time();
		return this.frames[this.get_current_frame_num()];
	}

	clamp_to_visible_region(pt) {
		return new Point2D(clamp(pt.x, this.visible_image_region.bmin.x, this.visible_image_region.bmax.x),
						   clamp(pt.y, this.visible_image_region.bmin.y, this.visible_image_region.bmax.y));
	}

	// convert point in canvas pixel coordinates to normalized [0,1]^2 image space coordinates
	canvas_to_image(pt) {

		var cur_frame = this.get_current_frame();

		var visible_box = this.visible_image_region.scale(cur_frame.source_image.width, cur_frame.source_image.height);
		var display_box = this.compute_image_display_box();

		// pixel space coordinates
		var image_pixel_x = visible_box.bmin.x + visible_box.width * (pt.x - display_box.bmin.x) / display_box.width;
		var image_pixel_y = visible_box.bmin.y + visible_box.height * (pt.y - display_box.bmin.y) / display_box.height;

		// normalized image space coordinates 
		var norm_x = clamp(image_pixel_x / cur_frame.source_image.width, 0.0, 1.0);
		var norm_y = clamp(image_pixel_y / cur_frame.source_image.height, 0.0, 1.0);

		return new Point2D(norm_x, norm_y);
	}

	// convert point in normalized [0,1]^2 image space coordinates to canvas pixel coordinates
	image_to_canvas(pt) {
		
		var cur_frame = this.get_current_frame();

		var visible_box = this.visible_image_region.scale(cur_frame.source_image.width, cur_frame.source_image.height);
		var display_box = this.compute_image_display_box();

		// pixel space coordinates of pt
		var image_pixel_x = pt.x * cur_frame.source_image.width;
		var image_pixel_y = pt.y * cur_frame.source_image.height;

		// convert into normalized coordinates in the visible region
		var visible_region_x = (image_pixel_x - visible_box.bmin.x) / visible_box.width;
		var visible_region_y = (image_pixel_y - visible_box.bmin.y) / visible_box.height;

		var display_x = display_box.bmin.x + visible_region_x * display_box.width;
		var display_y = display_box.bmin.y + visible_region_y * display_box.height;

		return new Point2D(display_x, display_y);
	}

	// true if the mouse is hovering over the canvas
	is_hovering() {
		return (this.cursorx >= 0 && this.cursory >= 0);
	}

	is_annotation_mode_per_frame_category() {
		return this.annotation_mode === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY;
	}

	is_annotation_mode_point() {
		return this.annotation_mode === Annotation.ANNOTATION_MODE_POINT;
	}

	is_annotation_mode_two_point_bbox() {
		return this.annotation_mode === Annotation.ANNOTATION_MODE_TWO_POINTS_BBOX;
	}

	is_annotation_mode_extreme_points_bbox() {
		return this.annotation_mode === Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX;
	}

	// Returns the index of the annotation that is the "selected annotation" given
	// the current mouse position
	// Per-frame annotations cannot be "selected" since there is only one per frame.
	// The main purpose of box/point annotation selection is to delete an annotation
	// and a per-frame annotation can be changed/deleted in a single keypress.  
	get_selected_annotation() {

		var selected = -1;

		if (!this.is_hovering())
			return selected;

		var image_cursor_pt = this.canvas_to_image(new Point2D(this.cursorx, this.cursory));

		var cur_frame = this.get_current_frame();

		// the cursor may fall within multiple boxes. Select the smallest (area) annotation
		// the cursor falls within.  The decision prevents a big box from preventing the selection of 
		// small ones.  It is possible that many small boxes could in agregate entirely cover a
		// big box, preventing the big box's selection, but this is far less common.
		var smallest_area = Number.MAX_VALUE;
		for (var i=0; i<cur_frame.data.annotations.length; i++) {
			
			if (cur_frame.data.annotations[i].type === Annotation.ANNOTATION_MODE_POINT) {

				if (image_cursor_pt.x === cur_frame.data.annotations[i].x &&
					image_cursor_pt.y === cur_frame.data.annotations[i].y) {
					selected = i;
					smallest_area = 0.0;
				}

			} else if (cur_frame.data.annotations[i].type === Annotation.ANNOTATION_MODE_TWO_POINTS_BBOX ||
					   cur_frame.data.annotations[i].type === Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX) {

				if (cur_frame.data.annotations[i].bbox.inside(image_cursor_pt) &&
					cur_frame.data.annotations[i].bbox.area < smallest_area) {
					selected = i;
					smallest_area = cur_frame.data.annotations[i].bbox.area;
				}
			}
		}

		return selected;			
	}

	clear_in_progress_points() {
		this.in_progress_points = [];
	}

	clear_zoom_corner_points() {
		this.zoom_corner_points = [];
	}

    has_per_frame_category_annotation() {
		var cur_frame = this.get_current_frame();

		// if a per-frame annotation exists, remove it
		for (var i=0; i<cur_frame.data.annotations.length; i++) {
			if (cur_frame.data.annotations[i].type === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
                return true;
			}
		}

        return false;
    }

	// FIXME(kayvonf): unify this with add_annotation() just like there's a common
	// interface for delete_annotation()
	set_per_frame_category_annotation(cat_idx) {

		var cur_frame = this.get_current_frame();

		// if a per-frame annotation exists, remove it
		for (var i=0; i<cur_frame.data.annotations.length; i++) {
			if (cur_frame.data.annotations[i].type === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
				cur_frame.data.annotations.splice(i, 1);
			}
		}

		var newAnnotation = new PerFrameAnnotation(cat_idx);
		if (cat_idx !== Annotation.INVALID_CATEGORY) {
			cur_frame.data.annotations.push(newAnnotation);
			//console.log("KLabeler: set per-frame annotation: " + this.category_to_name[cat_idx]);
			this.add_annotation(newAnnotation);
		}

		// will need to add an annotation_changed_callback
		//if (this.annotation_changed_callback !== null)
		//	this.annotation_changed_callback();
	}

	add_annotation(ann) {
		var cur_frame = this.get_current_frame()
		cur_frame.data.annotations.push(ann);
		if (this.annotation_added_callback !== null)
			this.annotation_added_callback(cur_frame, ann);
		if (this.annotation_changed_callback !== null)
			this.annotation_changed_callback();
	}

	delete_annotation() {
		console.log("Deleting annotation")

		var cur_frame = this.get_current_frame();

		// handle per-frame annotations a little differently since they cannot be selected
		// just delete the annotation on the frame is the labeler is in per-frame annotation mode
		if (this.is_annotation_mode_per_frame_category()) {

			if (cur_frame.image_load_complete)
				this.set_per_frame_category_annotation(Annotation.INVALID_CATEGORY);

		// for all other annotations, the user must select them to delete them
		} else {

			var selected = this.get_selected_annotation();

			if (selected !== -1) {
				const ann = cur_frame.data.annotations[selected];
				cur_frame.data.annotations.splice(selected, 1);
				//console.log("KLabeler: Deleted box " + selected);

				if (this.annotation_deleted_callback !== null)
					this.annotation_deleted_callback(cur_frame, ann)
				if (this.annotation_changed_callback !== null)
					this.annotation_changed_callback();
			}
		}

		this.render();
	}

	play_click_audio() {

		// if there are no in progress points, then this must be the last click needed to
		// make an annotation play the sound corresponding to a finished annotation. 
		if (this.in_progress_points.length === 0) {

			// stop other sounds
			this.audio_click_sound.pause();

			// play the end of box sound

			// if the sound is aready playing then the subsequent call to play() will do nothing
			// and the audio will keep playing from that point it is currently at.
			// (Or if the audio is paused, play() will resume from the paused point)
			// So I reset the playpack point of the sound to the start of the timeline so that
			// the sound plays again from the start.
			this.audio_box_done_sound.currentTime = 0.0;					
			this.audio_box_done_sound.play();

		} else {

			// stop other sounds
			this.audio_box_done_sound.pause();

			// play the click sound
			this.audio_click_sound.currentTime = 0.0;					
			this.audio_click_sound.play();
		}
	}

	// computes the canvas-space bounding box of the rendered image
	compute_image_display_box() {

		// visible region of the image in the image's pixel space

		var cur_frame = this.get_current_frame();
		var visible_box = this.visible_image_region.scale(cur_frame.source_image.width, cur_frame.source_image.height); 

		// by default: scale the image to fill the entire canvas
		var display_startx = 0;
		var display_starty = 0;
		var display_width = this.main_canvas_el.width;
		var display_height = this.main_canvas_el.height;;

		if (this.letterbox_image) {

			var aspect_canvas = this.main_canvas_el.height / this.main_canvas_el.width;
			var aspect_visible  = visible_box.height / visible_box.width;

			if (aspect_canvas >= aspect_visible) {
				// canvas is taller than the visible part of the image, so letterbox the top and bottom
				display_width = this.main_canvas_el.width;
				display_height = this.main_canvas_el.width * aspect_visible;
				display_startx = 0; 
				display_starty = (this.main_canvas_el.height - display_height) / 2;

			} else {
				// canvas is wider than the visible part of the image, so letterbox the left and right
				display_height = this.main_canvas_el.height;
				display_width = this.main_canvas_el.height / aspect_visible;
				display_startx = (this.main_canvas_el.width - display_width) / 2;
				display_starty = 0; 
			}
		}

		return new BBox2D(display_startx, display_starty, display_width, display_height);
	}

	draw_inprogress_extreme_points_bbox(ctx, canvas_in_progress_points, canvas_cursor_pt) {

		// draw lines between the points we've locked down
		ctx.beginPath();
		ctx.moveTo(canvas_in_progress_points[0].x, canvas_in_progress_points[0].y);
		for (var i=1; i<canvas_in_progress_points.length; i++) {
			var cornerx = 0;
			var cornery = 0;
			if (i === 1) {
				cornerx = canvas_in_progress_points[0].x;
				cornery = canvas_in_progress_points[1].y;
			} else if (i === 2) {
				cornerx = canvas_in_progress_points[2].x;
				cornery = canvas_in_progress_points[1].y;
			} else if (i === 3) {
				cornerx = canvas_in_progress_points[0].x;
				cornery = canvas_in_progress_points[3].y;
			}
			ctx.lineTo(cornerx, cornery);
			ctx.lineTo(canvas_in_progress_points[i].x, canvas_in_progress_points[i].y);
		}

		// now draw lines to the tentative point
		if (this.is_hovering()) {
			if (canvas_in_progress_points.length === 1) {
				ctx.lineTo(canvas_in_progress_points[0].x, canvas_cursor_pt.y);		
				ctx.lineTo(canvas_cursor_pt.x, canvas_cursor_pt.y);
			} else if (canvas_in_progress_points.length === 2) {
				ctx.lineTo(canvas_cursor_pt.x, canvas_in_progress_points[1].y);
				ctx.lineTo(canvas_cursor_pt.x, canvas_cursor_pt.y);
			} else if (canvas_in_progress_points.length === 3) {
				ctx.lineTo(canvas_in_progress_points[2].x, canvas_cursor_pt.y);
				// extrapolation of rest of box
				ctx.lineTo(canvas_in_progress_points[0].x, canvas_cursor_pt.y);
				ctx.lineTo(canvas_in_progress_points[0].x, canvas_in_progress_points[0].y);
			}
		}
		ctx.stroke();

		// draw dots at all the extreme points that have been specified so far
		var full_circle_angle = 2 * Math.PI;
		ctx.fillStyle = this.color_extreme_point_fill;
		for (i = 0; i < canvas_in_progress_points.length; i++) {
			ctx.beginPath();
				ctx.arc(canvas_in_progress_points[i].x, canvas_in_progress_points[i].y, this.extreme_point_radius, 0, full_circle_angle, false);
	        ctx.fill();
		}	
	}

	draw_inprogress_two_points_bbox(ctx, canvas_in_progress_points, canvas_cursor_pt) {

		var pts = [];
		pts.push(canvas_in_progress_points[0]);
		pts.push(canvas_cursor_pt);

		var box = BBox2D.two_points_to_bbox(pts);
		ctx.strokeRect(box.bmin.x, box.bmin.y, box.width, box.height);
	}

	draw_inprogress_zoom_bbox(ctx, canvas_zoom_corner_points, canvas_cursor_pt) {

		var pts = [];
		pts.push(canvas_zoom_corner_points[0]);
		pts.push(canvas_cursor_pt);

		var box = BBox2D.two_points_to_bbox(pts);

		//console.log("Drawing zoom box: " + box.to_string())

		ctx.fillStyle = this.color_zoom_box_fill;
		ctx.fillRect(box.bmin.x, box.bmin.y, box.width, box.height);

		ctx.lineWidth = 2;
		ctx.strokeStyle = this.color_zoom_box_outline;
		ctx.strokeRect(box.bmin.x, box.bmin.y, box.width, box.height);
	}

	draw_heatmap(ctx, cur_frame) {
		const PATCH_SIZE = 32;

		for (let i = 0; i < cur_frame.data.spatial_dists.length; i++) {
			let patch_idx = cur_frame.data.spatial_dists[i][0];
			let n_cols = Math.ceil(cur_frame.source_image.width / PATCH_SIZE);
			let x = (patch_idx % n_cols) * PATCH_SIZE;
			let y = Math.floor(patch_idx / n_cols) * PATCH_SIZE;

			let canvas_start = this.image_to_canvas(
				new Point2D(x / cur_frame.source_image.width, y / cur_frame.source_image.height)
			);
			let canvas_end = this.image_to_canvas(
				new Point2D((x + PATCH_SIZE) / cur_frame.source_image.width, (y + PATCH_SIZE) / cur_frame.source_image.height)
			);

			ctx.fillStyle = 'rgba(230,0,0,0.4)';
			ctx.fillRect(canvas_start.x, canvas_start.y, canvas_start.x - canvas_end.x, canvas_start.y - canvas_end.y);
		}
	}

	render() {

		var ctx = this.main_canvas_el.getContext('2d');

		ctx.fillStyle = this.color_background;
		ctx.fillRect(0, 0, this.main_canvas_el.width, this.main_canvas_el.height);
		
        if (this.frames.length === 0) {
          return;
        }
		var cur_frame = this.get_current_frame();

		//
		// draw the image being labeled
		//
		//console.log(cur_frame.image_load_complete)
		//console.log(cur_frame.source_image.onload)
		if (cur_frame.image_load_complete) {

			var visible_box = this.visible_image_region.scale(cur_frame.source_image.width, cur_frame.source_image.height);
			var display_box = this.compute_image_display_box();

			ctx.drawImage(cur_frame.source_image,
				visible_box.bmin.x, visible_box.bmin.y, visible_box.width, visible_box.height,
				display_box.bmin.x, display_box.bmin.y, display_box.width, display_box.height);

			if (cur_frame.data.spatial_dists !== undefined && cur_frame.data.spatial_dists !== null) {
				this.draw_heatmap(ctx, cur_frame);
			}
		}

		var image_cursor_pt = this.clamp_to_visible_region(this.canvas_to_image(new Point2D(this.cursorx, this.cursory)));
		var canvas_cursor_pt = this.image_to_canvas(image_cursor_pt);

		//
		// draw guidelines that move with the mouse cursor
		//

		if (this.show_crosshairs && this.is_hovering()) {
			ctx.lineWidth = 1;
			ctx.strokeStyle = this.color_cursor_lines;

			ctx.beginPath();
			ctx.moveTo(canvas_cursor_pt.x, 0);
			ctx.lineTo(canvas_cursor_pt.x, this.main_canvas_el.height);
			ctx.stroke();

			ctx.beginPath();
			ctx.moveTo(0, canvas_cursor_pt.y);
			ctx.lineTo(this.main_canvas_el.width, canvas_cursor_pt.y);
			ctx.stroke();
		}

		//
		// draw existing annotations.  These annotations may be bounding boxes, points,
		// or an annotation on the whole frame 
		//

		var selected = this.get_selected_annotation();

		for (var ann_index=0; ann_index<cur_frame.data.annotations.length; ann_index++) {

			var ann = cur_frame.data.annotations[ann_index];
			var is_selected = (selected === ann_index);

			// draw a point annotation
			if (ann.type === Annotation.ANNOTATION_MODE_POINT) {

				// do not render points that lie outside the visible region (when zoomed)
				if (!this.visible_image_region.inside(ann.pt))
					continue;

				var full_circle_angle = 2 * Math.PI;
				var canvas_pt = this.image_to_canvas(ann.pt);

				if (is_selected) {
					ctx.fillStyle = this.color_selected_point_fill;
				} else {
					ctx.fillStyle = this.color_point_fill;						
				}
				ctx.beginPath();
  				ctx.arc(canvas_pt.x, canvas_pt.y, this.extreme_point_radius, 0, full_circle_angle, false);
		        ctx.fill();

		    // draw bounding box annotation
			} else if (ann.type === Annotation.ANNOTATION_MODE_TWO_POINTS_BBOX ||
	   				   ann.type === Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX)  {

				// clip the bbox to the visible region of the image (if zoomed)
				var visible_ann_box = ann.bbox.intersect(this.visible_image_region);

				// if the annotation bbox doesn't overlap the visible region,
				// there's nothing to draw 
				if (!visible_ann_box.is_empty()) {

					// transform to canvas space
					var canvas_min = this.image_to_canvas(visible_ann_box.bmin);
					var canvas_max = this.image_to_canvas(visible_ann_box.bmax);
					var canvas_width = canvas_max.x - canvas_min.x;
					var canvas_height = canvas_max.y - canvas_min.y; 

					// highlight the selected box
					if (is_selected) {
						ctx.lineWidth = 3;
						ctx.strokeStyle = this.color_selected_box_outline;
						ctx.fillStyle = this.color_selected_box_fill;
						ctx.fillRect(canvas_min.x, canvas_min.y, canvas_width, canvas_height);
					} else {
						ctx.lineWidth = 2;
						ctx.strokeStyle = this.color_box_outline;
					}

					ctx.strokeRect(canvas_min.x, canvas_min.y, canvas_width, canvas_height);
				}

				// if this is a box created from extreme points, draw dots indicating all the extreme points
				if (this.show_extreme_points && ann.type === Annotation.ANNOTATION_MODE_EXTREME_POINTS_BBOX)  {
					full_circle_angle = 2 * Math.PI;
					ctx.fillStyle = this.color_extreme_point_fill;
					for (var i=0; i<4; i++) {

						// do not render extreme points that lie outside the visible region (when zoomed)
						if (!this.visible_image_region.inside(ann.extreme_points[i])) {
							continue;
						}

						canvas_pt = this.image_to_canvas(ann.extreme_points[i]);
						ctx.beginPath();
	      				ctx.arc(canvas_pt.x, canvas_pt.y, this.extreme_point_radius, 0, full_circle_angle, false);
				        ctx.fill();
					}
				}	
			} else if (ann.type === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {

                var line_width = 14;
				// draw box around the display in the appropriate color to indicate the per-frame label
				ctx.strokeStyle = this.category_to_color[ann.value];
				ctx.lineWidth = line_width;
			    ctx.strokeRect(line_width/2, line_width/2,
                               this.main_canvas_el.width-line_width,
                               this.main_canvas_el.height-line_width);

				ctx.fillStyle = this.color_category_text_fill;
				ctx.font = this.category_text_font; 
				ctx.fillText(this.category_to_name[ann.value], 4, this.main_canvas_el.height - 5);
			}
		}

		//
		// draw "in-progress" points (e.g., the current partial bounding box)
		//

		if (this.in_progress_points.length > 0) {

			// convert image-space points to canvas space for drawing on screen
			var canvas_in_progress_points = [];
			for (i = 0; i<this.in_progress_points.length; i++)
				canvas_in_progress_points[i] = this.image_to_canvas(this.in_progress_points[i]);

			ctx.lineWidth = 1;
			ctx.strokeStyle = this.color_in_progress_box_outline; 

			if (this.is_annotation_mode_extreme_points_bbox()) {
				this.draw_inprogress_extreme_points_bbox(ctx, canvas_in_progress_points, canvas_cursor_pt);
			} else if (this.is_annotation_mode_two_point_bbox()) {
				this.draw_inprogress_two_points_bbox(ctx, canvas_in_progress_points, canvas_cursor_pt);
			}
		}

		//
		// draw zoom box UI
		//

		if (this.zoom_corner_points.length > 0) {

			// convert image-space points to canvas space for drawing on screen
			var canvas_zoom_corner_points = [];
			for (i = 0; i < this.zoom_corner_points.length; i++)
				canvas_zoom_corner_points[i] = this.image_to_canvas(this.zoom_corner_points[i]);

			this.draw_inprogress_zoom_bbox(ctx, canvas_zoom_corner_points, canvas_cursor_pt);
		}

	}

	handle_image_load(source_url, image_index) {
		//console.log("KLabeler: Image " + image_index + " loaded.");
        if (image_index < this.frames.length && this.frames[image_index].data.source_url === source_url) {
		  this.frames[image_index].image_load_complete = true;
		  this.render();
        }
	}

	handle_canvas_mousemove = event => {
		this.set_canvas_cursor_position(event.offsetX, event.offsetY);
		this.render();
	}

	handle_canvas_mouseover = event => {
		this.set_canvas_cursor_position(event.offsetX, event.offsetY);		
		this.render();
	}

	handle_canvas_mouseout = event => {
		this.cursorx = Number.MIN_SAFE_INTEGER;
		this.cursory = Number.MIN_SAFE_INTEGER;

		// NOTE(kayvonf): decision made to not clear at this time since mouse can
		// often leave the canvas as a result of user motion just to find the cursor.
		//this.clear_zoom_corner_points();
		//this.clear_in_progress_points();

		this.render();
	}

	stop_autolabel(){
		if (this.autoStepId) {
			clearInterval(this.autoStepId)
			this.autoStepId = null;
		}
	}

	handle_keydown(event, prevFrame, nextFrame) {
		console.log("KeyDown: " + event.keyCode);
		// number key: 0-9
		var stopAuto = (this.current_frame_index === nextFrame);
		if (event.keyCode >= 48 && event.keyCode <= 57) {
			var key_pressed = event.keyCode - 48;
			var category_name = this.category_to_name[key_pressed];

			// ignore keys if image hasn't loaded yet
			var cur_frame = this.get_current_frame();
			if (cur_frame.image_load_complete && category_name !== "") {
				this.set_per_frame_category_annotation(key_pressed);
			}
			//this.render();
			this.current_indices = [nextFrame]
			setTimeout(() => {this.set_current_frame_num(nextFrame)},100);
		} else if (event.keyCode === 65) {   // "a" for autolabel
			if (!this.autoStepId) {
				var timeDiv = document.getElementById("frameSpeed")
				var imageTime = 500;
				if (timeDiv) {
					var divValue = parseInt(timeDiv.value)
					if (!isNaN(divValue)) {
						imageTime = divValue
					}
					if (imageTime < 250) {
						imageTime = 250
					}
				}
				this.autoStepId = setInterval(() => {
					console.log("Keypress")
					var evt = new KeyboardEvent("keydown", {
					bubbles : true,
					cancelable : true,
					char : "2",
					key : "2",
					shiftKey : false,
					keyCode : 50
					})
					console.log(evt)
					window.dispatchEvent(evt);
				}, imageTime)
			}
		} else if (event.keyCode === 83) {   // "s" for autolabel stop
			this.stop_autolabel();
		} else if (event.keyCode === 68) {   // "d" for autolabel defer (marking last few frames unsure)
			// if a per-frame annotation exists, remove it
			if (this.autoStepId) {
				var timeDiv = document.getElementById("frameSpeed")
				var imageTime = 500;
				if (timeDiv) {
					var divValue = parseInt(timeDiv.value)
					if (!isNaN(divValue)) {
						imageTime = divValue
					}
					if (imageTime < 250) {
						imageTime = 250
					}
				}
				// label last 2 seconds of images as unsure
				var j = 0;
				var currFrameNum = this.get_current_frame_num();
				while (imageTime*j < 2000 && j <= currFrameNum) {
					cur_frame = this.frames[currFrameNum - j];
					for (var i=0; i<cur_frame.data.annotations.length; i++) {
						if (cur_frame.data.annotations[i].type === Annotation.ANNOTATION_MODE_PER_FRAME_CATEGORY) {
							cur_frame.data.annotations.splice(i, 1);
						}
					}
					var newAnnotation = new PerFrameAnnotation(3); // Unsure
					cur_frame.data.annotations.push(newAnnotation);
					//console.log("KLabeler: set per-frame annotation: " + this.category_to_name[cat_idx]);
					this.add_annotation(newAnnotation);
					j += 1
				}
			}
		}
		if (stopAuto) {
			this.stop_autolabel();
		}

		if (event.keyCode === 37) {   // left arrow
			if (this.current_frame_index > 0) {
				this.current_indices = [prevFrame]
				this.set_current_frame_num(prevFrame);
			} else {
				this.update_labeling_time();
			}

			// reset the zoom
			if (!this.retain_zoom) {
				this.visible_image_region = new BBox2D(0.0, 0.0, 1.0, 1.0);
				this.render();
			}
			this.stop_autolabel();
		} else if (event.keyCode === 39) {  // right arrow
			console.log("Handling right arrow")
			if (this.current_frame_index < this.frames.length-1) {
				this.current_indices = [nextFrame]
				this.set_current_frame_num(nextFrame);
			} else {
				this.update_labeling_time();
			}

			// reset the zoom
			if (!this.retain_zoom) {
				this.visible_image_region = new BBox2D(0.0, 0.0, 1.0, 1.0);
				this.render();
			}
			this.stop_autolabel();
		} else if (event.keyCode === 27) {  // ESC key

			this.clear_in_progress_points();
			this.clear_zoom_corner_points();
			this.render();
			this.stop_autolabel();
		} else if (event.keyCode === 82) {  // 'r' key: reset zoom
			this.visible_image_region = new BBox2D(0.0, 0.0, 1.0, 1.0);
			this.render();

		} else if (event.keyCode === 90) {  // 'z' key 
			if (!this.zoom_key_down) {
				this.zoom_key_down = true;
				this.clear_zoom_corner_points();
				this.clear_in_progress_points();
			}
			this.render();
		}
	}

	handle_keyup = event => {

		if (event.keyCode === 8) {            // delete/bksp key
			this.delete_annotation();

		} else if (event.keyCode === 90) {   // 'z' key (zoom mode)
			this.zoom_key_down = false;
			this.clear_zoom_corner_points();
		}

		this.render();
	}

	handle_canvas_click = event => {
		this.set_focus();
		var cur_frame = this.get_current_frame();

		// ignore mouse clicks if the image hasn't loaded yet
		if (!cur_frame.image_load_complete)
			return;

		this.set_canvas_cursor_position(event.offsetX, event.offsetY);
		var image_cursor_pt = this.clamp_to_visible_region(this.canvas_to_image(new Point2D(this.cursorx, this.cursory)));

		// if the user is holding down the "zoom mode key", treat the click as specifying a
		// zoom bounding box, not as defining an annotation.
		if (this.zoom_key_down) {

			this.zoom_corner_points.push(image_cursor_pt);

			if (this.zoom_corner_points.length === 2) {
				this.visible_image_region = BBox2D.two_points_to_bbox(this.zoom_corner_points);
				this.clear_zoom_corner_points(); 
				//console.log("Set visible region: (" + this.visible_image_region.bmin.x + ", " + this.visible_image_region.bmin.y + "), w=" + this.visible_image_region.width + ", h=" + this.visible_image_region.height);
			}

			this.render();
			return;
		}

		this.in_progress_points.push(image_cursor_pt);		
		//console.log("KLabeler: Click at (" + this.cursorx + ", " + this.cursory + "), image space=(" + image_cursor_pt.x + ", " + image_cursor_pt.y + "), point " + this.in_progress_points.length);

		if (this.is_annotation_mode_extreme_points_bbox() && this.in_progress_points.length === 4) {

			// discard box if this set of four extreme points is not a valid set of extreme points
			if (!BBox2D.validate_extreme_points(this.in_progress_points)) {
				console.log("KLabeler: Points clicked are not valid extreme points. Discarding box.");
				this.clear_in_progress_points();
				this.render();
				return;
			}

			var new_annotation = new ExtremeBoxAnnnotation(this.in_progress_points);
			this.add_annotation(new_annotation);
			//console.log("KLabeler: New box: x=[" + new_annotation.bbox.bmin.x + ", " + new_annotation.bbox.bmax.x + "], y=[" + new_annotation.bbox.bmin.y + ", " + new_annotation.bbox.bmax.y + "]");

			this.clear_in_progress_points();

		// this click completes a new corner point box annotation
		} else if (this.is_annotation_mode_two_point_bbox() && this.in_progress_points.length === 2) {

			// validate box by discarding empty boxes.
			if (this.in_progress_points[0].x === this.in_progress_points[1].x &&
				this.in_progress_points[0].y === this.in_progress_points[1].y) {
				alert("Empty bbox. Discarding box.");
				this.clear_in_progress_points();
				this.render();
				return;
			}

			new_annotation = new TwoPointBoxAnnotation(this.in_progress_points);
			this.add_annotation(new_annotation);
			//console.log("KLabeler: New box: x=[" + new_annotation.bbox.bmin.x + ", " + new_annotation.bbox.bmax.y + "], y=[" + new_annotation.bbox.bmin.y + ", " + new_annotation.bbox.bmax.y + "]");

			this.clear_in_progress_points();

		// this click completes a new point annotation
		} else if (this.is_annotation_mode_point()) {

			new_annotation = new PointAnnotation(this.in_progress_points[0]);
			this.add_annotation(new_annotation);
			//console.log("KLabeler: New point: (" + new_annotation.pt.x + ", " + new_annotation.pt.y + ")");

			this.clear_in_progress_points();
		}

		this.render();

		if (this.play_audio)
			this.play_click_audio();
	}

	/////////////////////////////////////////////////////////////////////////////////////////////
	// The following methods constitute KLabeler's application-facing API
	// (called by driving applications)
	/////////////////////////////////////////////////////////////////////////////////////////////

	set_focus() {
		this.main_canvas_el.focus();
	}

	set_background_color(color) {
		this.color_background = color;
	}

	set_retain_zoom(value) {
		this.retain_zoom = value;
	}

	clear_boxes() {
		var cur_frame = this.get_current_frame();

		cur_frame.data.annotations = [];
		this.clear_in_progress_points();
		this.clear_zoom_corner_points();
		this.render();
	}

	set_annotation_mode(mode) {
		this.annotation_mode = mode;
		this.clear_in_progress_points();
		this.clear_zoom_corner_points();
		this.render();
	}

	set_categories(categories) {

		// modifies key bindings in per-frame categorical mode
		this.category_to_name = Array(10).fill("");
		this.category_to_color = Array(10).fill("");

		Object.entries(categories).forEach( entry => {
			var category_name = entry[0];
			var category_key = entry[1].value;
			var category_color = entry[1].color;
			this.category_to_name[category_key] = category_name;
			this.category_to_color[category_key] = category_color;
		});
	}

	set_extreme_points_viz(status) {
		this.show_extreme_points = status;
		this.render();
	}

	set_play_audio(toggle) {
		this.play_audio = toggle;
	}

	set_crosshairs_viz(value) {
		this.show_crosshairs = value;
		this.render();
	}

	set_letterbox(toggle) {
		this.letterbox_image = toggle;
		this.clear_in_progress_points();
		this.clear_zoom_corner_points();
		this.render();
	}

	get_current_frame_num() {
		return this.current_frame_index;
	}

	get_num_frames() {
		return this.frames.length;
	}

	set_current_frame_num(frame_num) {

		this.update_labeling_time();

		this.current_frame_index = frame_num;
		//this.current_indices = [frame_num];
		this.clear_in_progress_points();
		this.clear_zoom_corner_points();
		this.render();

		if (this.frame_changed_callback !== null)
			this.frame_changed_callback(this.current_frame_index);
	}

	set_frame_changed_listener(func) {
		this.frame_changed_callback = func;
	}

	set_annotation_changed_listener(func) {
		this.annotation_changed_callback = func;
	}

	make_image_load_handler(source_url, x) {
		return event => {
			this.handle_image_load(source_url, x);
		}
	}

	load_image_stack(image_dataset) {
		console.log('KLabeler: loading set of ' + image_dataset.length + ' images.');
		this.stop_autolabel();
        if (this.frames.length > 0) {
		  this.set_current_frame_num(0);
        }

		this.frames = [];
		var image_index = 0;
		for (var img of image_dataset) {
			var frame = new Frame(img);

			// kick off the image load
			frame.image_load_started = true;
			frame.source_image.onload = this.make_image_load_handler(img.source_url, image_index);
			frame.source_image.src = frame.data.source_url;
			this.frames.push(frame);
			image_index++;
		}

		// FIXME(kayvonf): extract to helper function
		// reset the viewer sequence
		this.visible_image_region = new BBox2D(0.0, 0.0, 1.0, 1.0);
		this.clear_in_progress_points();
		this.clear_zoom_corner_points();
        if (this.frames.length > 0) {
		  this.set_current_frame_num(0);
		}
		this.current_indices = [0];
	}

	get_annotations() {

		this.update_labeling_time();
		
		var results = [];
		for (var i=0; i<this.frames.length; i++) {
			results.push(this.frames[i].data);
		}

		return results;
	}

	get_current_frame_data() {
		return this.get_current_frame().data
	}

	init(main_canvas_el) {

		console.log("Klabeler: initializing...");

		this.main_canvas_el = main_canvas_el;
		this.main_canvas_el.addEventListener("mousemove", this.handle_canvas_mousemove, false);
		this.main_canvas_el.addEventListener("click", this.handle_canvas_click, false);
		this.main_canvas_el.addEventListener("mouseover", this.handle_canvas_mouseover, false);
		this.main_canvas_el.addEventListener("mouseout", this.handle_canvas_mouseout, false);

		//this.main_canvas_el.addEventListener("keydown", this.handle_keydown, false);
		//this.main_canvas_el.addEventListener("keyup", this.handle_keyup, false);

		// make a dummy frame as a placeholder until the application provides real data
		this.frames.push(new Frame(new ImageData()));
		
		// FIXME(kayvonf): extract to helper function
		// reset the viewer sequence
		this.visible_image_region = new BBox2D(0.0, 0.0, 1.0, 1.0);
		this.clear_in_progress_points();
		this.clear_zoom_corner_points();
		this.set_current_frame_num(0);
	}
}
