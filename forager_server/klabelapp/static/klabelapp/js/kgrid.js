
class KGridOptions {
	constructor() {

		// a panel is one-to-one with an image thumbnail
		this.panel_width = 200;
		this.panel_height = 200;

		// each "cell" of the grid may contain multiple panels
		// for example, to recreate the visualization of an image beside its heatmap,  
		// set this value to 2 and provide each cell the appropriate two image URLs
		this.num_panels = 1;

		// how to display the image thumbnail in the panel.  If false, scale to fill the entire panel 
		this.letterbox_panels = false;

		// cells can either be arranged in a 2D grid, or a single cell per row.
		// This option is true if the layout is a single cell per row.
		this.single_cell_per_row = false;
	}
}

class KGridCell {
	constructor(image_urls, text) {
		this.image_urls = image_urls;
		this.text = text;
	}
}

class KGrid {
	constructor() {
		this.containing_div = null;
	}

	init(div_el, options) {
		this.containing_div = div_el;
		this.options = options;
	}

	make_image_load_handler(img_el) {
		return event => {
			console.log('Loaded image: ' + img_el.src);
		}
	}

	make_image_click_handler(img_el) {
		return event => {
			console.log('Clicked image: ' + img_el.src);
		}
	}

	generate_grid(cell_data) {
		
		for (var i=0; i<cell_data.length; i++) {

			// validate source data
			if (cell_data[i].image_urls.length != this.options.num_panels) {
				console.error('KGrid: cell ' + i + ' data has incorrect number of image urls (' + cell_data[i].image_urls.length + ')');
				return;
			}

			var div_el = document.createElement('div');

			var text_el = document.createElement('div');
			var text_node = document.createTextNode(cell_data[i].text);
			
			text_el.appendChild(text_node);
			div_el.appendChild(text_el);

			for (var j=0; j<this.options.num_panels; j++) {

				var img_el = document.createElement('img');
				img_el.onload = this.make_image_load_handler(img_el);
				img_el.onclick = this.make_image_click_handler(img_el);
				img_el.src = cell_data[i].image_urls[j];
				img_el.style.width = "" + this.options.panel_width + "px";
				img_el.style.height = "" + this.options.panel_width + "px";
				img_el.classList.add("kgrid_image");
				img_el.label = 
				div_el.appendChild(img_el);
			}

			div_el.classList.add("kgrid_cell");
			if (!this.options.single_cell_per_row)
				div_el.classList.add("kgrid_cell_floatprop");

			this.containing_div.appendChild(div_el);
		}
	}

	get_labels() {


	}




}