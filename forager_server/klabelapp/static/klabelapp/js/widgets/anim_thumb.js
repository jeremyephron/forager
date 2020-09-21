class AnimatedThumbnailWidget {

	constructor() {
		this.dom_image = null;
		this.datapoint_idx = -1;
		this.current_frame_idx = -1;
		this.urls = null;
		this.anim_request = null;
		this.last_paint_time = null;
		this.delay = 0;
		this.owns_events = null;
	}

	// FIXME(kayvonf): the owns_events parameter exists because there are two ways this object
	// gets attached to an image: the main preview way and the small thumbnail way.
	// I will clean this up later.
	attach(dom_image, datapoint_idx, urls, owns_events) {

		if (this.dom_image != null)
			this.detach();

		this.dom_image = dom_image;
		this.urls = urls;
		this.datapoint_idx = datapoint_idx;

		// FIXME(kayvonf): I currently hardcoded index to '1' for IMAGE_URL_SEQ case
		this.current_frame_idx = 1;
	
		this.owns_events = owns_events;

		// prefetch for animation purposes
		this.images = [];
		for (var i=0; i<this.urls.length; i++)
			this.images[i] = new Image(this.urls[i]);

		if (this.owns_events) {
			this.dom_image.addEventListener("mouseover", this.handle_mouseover, false);
			this.dom_image.addEventListener("mouseout", this.handle_mouseout, false);
			this.start_animation(0);
		} else {
			this.start_animation(0);
		}
	}

	detach() {
		if (this.dom_image != null) {
			this.stop_animation();
			if (this.owns_events) {
				this.dom_image.removeEventListener("mouseover", this.handle_mouseover, false);
				this.dom_image.removeEventListener("mouseout", this.handle_mouseout, false);
			}
		}
		this.dom_image = null;
	}

	animate() {
		var current_time = performance.now();
		if (current_time - this.last_paint_time > this.delay) {
			this.delay = 100;
			this.current_frame_idx++;
			if (this.current_frame_idx == this.urls.length)
				this.current_frame_idx = 0;
			this.dom_image.src = this.urls[this.current_frame_idx];
			this.last_paint_time = current_time;
		}
		this.anim_request = requestAnimationFrame( () => this.animate());
	}

	start_animation(initial_delay) {
		
		// animation could be running, so stop and restart
		this.stop_animation();

		this.delay = initial_delay;
		this.last_paint_time = performance.now();
		this.anim_request = requestAnimationFrame( () => this.animate());
	}

	stop_animation() {
		cancelAnimationFrame(this.anim_request);
		
		// FIXME(kayvonf): I currently hardcoded index to '1' for IMAGE_URL_SEQ case
		this.current_frame_idx = 1;
		
		this.dom_image.src = this.urls[this.current_frame_idx];
	}

	handle_mouseover = event => {
		this.start_animation(0);
	}

	handle_mouseout = event => {
		this.stop_animation();
	}
}

