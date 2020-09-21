// this file depends on:
//   -- kmath.js


// TODO LIST:
//   -- a bunch of data in the dump is of type float and could be integer

class LFViz {

	constructor() {

		this.main_canvas_el = null;
		this.cached_canvas_image = null;

		this.data_preview_func = null;

		this.cursorx = Number.MIN_SAFE_INTEGER;
		this.cursory = Number.MIN_SAFE_INTEGER;

		// data
		this.num_rows = 0;
		this.num_lf = 0;
		this.data_matrix = null;

		// visualization
		this.row_filter_mask = null;
		this.row_sorting = null;
		this.preview_idx = -1;

		// selection
		this.cur_selection_idx = -1;

		// color constants
		this.color_main_canvas = '#e0e0e0';
		this.color_filtered_row = '#d0d0d0';
		this.color_lf_positive = '#67bf5c';
		this.color_lf_negative = '#ed665d';
		this.color_lf_abstain = '#a2a2a2';
		this.color_highlight_box_outline = 'rgba(0.0, 0.0, 0.0, 1.0)';

		// layout parameters
		this.display_highlight_line_width = 2;
		this.display_el_width = 7;
		this.display_el_height = 7;
		this.display_col_sep = this.display_el_width;

	}

	// true if the mouse is hovering over the canvas
	is_hovering() {
		return (this.cursorx >= 0 && this.cursory >= 0);
	}

	// true if there is a currently selected datapoint, false otherwise
	has_selection() {
		return (this.cur_selection_idx != -1);
	}

	// sets the datapoint that is currently being hovered over
	make_selection() {
		this.cur_selection_viz_idx = this.get_highlighted_viz_cell();
		this.cur_selection_idx = this.get_highlighted_datapoint();
	}

	clear_selection() {
		this.cur_selection_idx = -1;
		this.cur_selection_viz_idx = -1;
	}

	// Clamp the cursor to the image dimensions
	set_canvas_cursor_position(x,y) {
		this.cursorx = clamp(x, 0, this.main_canvas_el.width);
		this.cursory = clamp(y, 0, this.main_canvas_el.height);	
	}

	// Returns the index (in the visualizer) of the "cell" that is being hovered over.  
	// The top-left corner of the visualization is cell 0.
	// Keep in mind that because of row sorting, the cell is not necessarily the same 
	// as the row of the datapoint that is being hovered over.
	// To get the data row, use get_highilghted_viz_cell()  
	get_highlighted_viz_cell() {

		// first get the cursor's row
		var row = Math.floor(this.cursory / this.display_el_height);

		// then get the cursor's column
		var spaced_col_width = this.display_el_width*this.num_lf + this.display_col_sep;
		var col = Math.floor(this.cursorx / spaced_col_width);

		// check to see if cursor is within the extent of the column, or in the margin to the right?
		// NOTE(kayvonf): I removed this check because I thought the image "flash" that occurs when the
		// cursor goes into the margin and the highlight disappears was jarring
		// 
		//if (this.cursorx - (col * spaced_col_width) > (this.display_el_width*this.num_lf))
		//	return -1;

		// compute index
		var rows_per_col = Math.floor(this.main_canvas_el.height / this.display_el_height);
		return col * rows_per_col + row;
	}

	// returns the index of the datapoint that is being hovered over
	get_highlighted_datapoint() {

		var viz_row_idx = this.get_highlighted_viz_cell();

		// should be a valid row, and a row that is not filtered out
		if (viz_row_idx < this.num_rows && this.row_filter_mask[this.row_sorting[viz_row_idx]] == true)
			return this.row_sorting[viz_row_idx];
		else
			return -1;
	}

	// Render the matrix visualization and cache the results in an image.
	// Rendering many boxes can be expensive, so the point of this caching is to avoid
	// having to draw all the visual elements of the visualization on every mouse move.
	render_cached_viz() {

		var ctx = this.main_canvas_el.getContext('2d');

		ctx.fillStyle = this.color_main_canvas;
		ctx.fillRect(0, 0, this.main_canvas_el.width, this.main_canvas_el.height);

		var rows_per_col = Math.floor(this.main_canvas_el.height / this.display_el_height);
		var num_cols = Math.floor((this.num_rows + rows_per_col - 1) / rows_per_col); 

		var spaced_col_width = this.display_el_width * this.num_lf + this.display_col_sep;

		if (num_cols * spaced_col_width > this.main_canvas_el.width)
			console.log("KLFViz: WARNING: amount of data too large for canvas, not showing all data!");

		for (var col=0; col<num_cols; col++) {

			var start_row = col * rows_per_col;
			var end_row = Math.min(start_row + rows_per_col, this.num_rows);
			var rows_in_this_col = end_row - start_row;

			// now draw a column of data points
			for (var i=0; i<rows_in_this_col; i++) {
	
				var viz_row_idx = start_row + i;
				var row_idx = this.row_sorting[viz_row_idx];

				if (this.row_filter_mask[row_idx] == true) {
					var start_y = i*this.display_el_height;
					for (var j=0; j<this.num_lf; j++) {
						var idx = row_idx * this.num_lf + j;

						var el_color = this.color_lf_abstain;
						if (this.data_matrix[idx] == 1)
							el_color = this.color_lf_positive;
						else if (this.data_matrix[idx] == -1)
							el_color = this.color_lf_negative;

						var start_x = col * spaced_col_width + j*this.display_el_width;
						ctx.fillStyle = el_color;
						ctx.fillRect(start_x, start_y, this.display_el_width, this.display_el_height);
					}
				} else {
					var start_x = col * spaced_col_width;
					var start_y = i*this.display_el_height;
					var col_width = this.display_el_width * this.num_lf;
					ctx.fillStyle = this.color_filtered_row;
					ctx.fillRect(start_x, start_y, col_width, this.display_el_height);
				}
			}
		}

		// store off what was just rendered into an image so it does not need to be rerendered each frame
		this.cached_canvas_image = ctx.getImageData(0, 0, this.main_canvas_el.width, this.main_canvas_el.height);
	}

	// main rendering routine
	render() {

		var ctx = this.main_canvas_el.getContext('2d');

		ctx.fillStyle = this.color_main_canvas;
		ctx.fillRect(0, 0, this.main_canvas_el.width, this.main_canvas_el.height);

		// draw the cached image of the grid visualization (previously rendered)
		ctx.putImageData(this.cached_canvas_image, 0, 0);

		// If there is a current selection, highlight it.
		// If there is no current selection, check to see if the cursor is
		// hovering over a datapoint row. If so, highlight the row being hovered over.
		var idx = this.cur_selection_idx; 
		var viz_idx = this.cur_selection_viz_idx;

		if (this.cur_selection_idx == -1 && this.is_hovering()) {
			idx = this.get_highlighted_datapoint();
			viz_idx = this.get_highlighted_viz_cell(); 
		}

		// draw the highlight
		if (idx >= 0) {

			var rows_per_col = Math.floor(this.main_canvas_el.height / this.display_el_height);
			var row = viz_idx % rows_per_col;
			var col = Math.floor(viz_idx / rows_per_col);

			var spaced_col_width = this.display_el_width*this.num_lf + this.display_col_sep;

			ctx.lineWidth = this.display_highlight_line_width;
			ctx.strokeStyle = this.color_highlight_box_outline;

			for (var i=0; i<this.num_lf; i++) {
				ctx.strokeRect(col*spaced_col_width + this.display_el_width*i, row*this.display_el_height,
			    		   this.display_el_width, this.display_el_height);
			}
		}
	}

	// Updates the contents of the datapoint preview DIV.
	// Pretty print the LF and label model results and display the
	// source data (either a text string or an image)
	//
	// To avoid flicker as the mouse moves around, we only redraw the preview window if the
	// datapoint to put into the preview window is different from the datapoint already
	// visualized there.  force_redraw=true overrides this
	update_preview(force_redraw = false) {

		var idx;
		if (this.has_selection())
			idx = this.cur_selection_idx;
		else {
			idx = this.get_highlighted_datapoint();
		}

		if (idx >= 0) {

			if (idx != this.preview_idx || force_redraw) {
				this.preview_idx = idx;
				this.data_preview_func(idx);
			}

    	} else {
    		this.preview_idx = -1;
    		this.data_preview_func(-1);
		}
	}

	// Change the input data for the visualizer.
	// Currently, this is treated a hard reset of the state of the visualizer.
	// Updating the input data resets both the row filter mask and the row sorting
	// See 'update_data()' instead for retaining all viz state and just swapping out data values
	set_data(num_rows, num_lf, lf_matrix, model_scores, data_preview_func) {
		this.num_rows = num_rows;
		this.num_lf = num_lf;
		this.data_matrix = lf_matrix;
		this.model_scores = model_scores;
		this.data_preview_func = data_preview_func;

		// reset the filter mask
		this.row_filter_mask = [];
		for (var i=0; i<num_rows; i++) {
			this.row_filter_mask.push(true); 
		}

		// reset the sorting
		this.row_sorting = [];
		for (var i=0; i<num_rows; i++) {
			this.row_sorting.push(i);
		}

		// clear the selection
		this.clear_selection();

		console.log("KLFViz: loading data (num rows=" + this.num_rows + ", num lf=" + this.num_lf + ")");

		this.render_cached_viz();
		this.render();
		this.update_preview(true);
	}

	// update the LF matrix and probablistic label values with new values
	// maintain all other data of the visualizer
	update_data(lf_matrix, model_scores) {
		this.data_matrix = lf_matrix;
		this.model_scores = model_scores;
		this.render_cached_viz();
		this.render();
		this.update_preview(true);
	}

	get_selection() {
		return this.cur_selection_idx;
	}

	set_selection(idx) {
		this.cur_selection_idx = idx; 

		if (this.has_selection()) {
			for (var i=0; i<this.num_rows; i++)
				if (this.row_sorting[i] == this.cur_selection_idx)
					this.cur_selection_viz_idx = i;
		}

		if (this.has_selection() && this.row_filter_mask[this.cur_selection_idx] == false) {
			this.clear_selection();
			var error_str = "Not selecting datapoint " + idx + " since it's currently filtered. Remove filter before selecting."; 
			console.log(error_str);
			alert(error_str);
		}

		this.render();
		this.update_preview();
	}

	// The visualization will permute the datapoint according to the indices in row_sorting
	set_row_sorting(row_sorting) {
		this.row_sorting = row_sorting;

		// sorting changed, so the viz idx of the current selection now also changes
		// scan over the data to find the new position of the selected datapoint
		if (this.has_selection()) {
			for (var i=0; i<this.num_rows; i++)
				if (this.row_sorting[i] == this.cur_selection_idx)
					this.cur_selection_viz_idx = i;
		}

		this.render_cached_viz();
		this.render();
		this.update_preview();
	}

	// The row filter mask determines which datapoints get shown in the visualization.
	// Datapoints that aren't included in the filter are not drawn.
	set_row_filter_mask(row_filter_mask) {
		this.row_filter_mask = row_filter_mask;

		// if the filter mask filters out the selection, go ahead and clear the selection.
		// Otherwise it's confusing since we would not be displaying the selected datapoint.
		if (this.has_selection() && this.row_filter_mask[this.cur_selection_idx] == false)
			this.clear_selection();

		this.render_cached_viz();
		this.render();
		this.update_preview();
	}

	handle_canvas_mousemove = event => {
		this.set_canvas_cursor_position(event.offsetX, event.offsetY);
		if (!this.has_selection()) {
			this.render();
			this.update_preview();
		}
	}

	handle_canvas_mouseover = event => {
		this.set_canvas_cursor_position(event.offsetX, event.offsetY);		
		if (!this.has_selection()) {
			this.render();
			this.update_preview();
		}
	}

	handle_canvas_mouseout = event => {
		this.cursorx = Number.MIN_SAFE_INTEGER;
		this.cursory = Number.MIN_SAFE_INTEGER;
		this.render();
		this.update_preview();
	}

	handle_canvas_click = event => {

		var idx = this.get_highlighted_datapoint();

		// click on currently selected item to clear
		if (this.has_selection() && idx == this.get_selection()) {
			this.clear_selection();
		} else {
			this.make_selection();
		}

		this.render();
		this.update_preview();
	}

	init(main_canvas_el, canvas_width, canvas_height, box_width, box_height) {

		this.main_canvas_el = main_canvas_el;

		this.main_canvas_el.width = canvas_width;
		this.main_canvas_el.height = canvas_height;
		this.display_el_width = box_width;
		this.display_el_height = box_height;

		// for small LF boxes, want a thinner highlight so it is possible to still see the boxes.
		if (this.display_el_width < 6 || this.display_el_height < 6)
			this.display_highlight_line_width = 1;

		this.main_canvas_el.addEventListener("mousemove", this.handle_canvas_mousemove, false);
		this.main_canvas_el.addEventListener("mouseover", this.handle_canvas_mouseover, false);
		this.main_canvas_el.addEventListener("mouseout",  this.handle_canvas_mouseout,  false);
		this.main_canvas_el.addEventListener("click", this.handle_canvas_click, false);

		this.render_cached_viz();
		this.render();
	}
}
