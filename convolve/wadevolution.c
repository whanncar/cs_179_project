

float calculate_overlap_product(Matrix *input, Matrix *kernel, int x, int y) {
	int x_min, y_min, x_max, y_max;
	int i, j;
	float result;

	if (x + kernel->width <= 0) {
		return 0;
	}
	if (x >= input->width) {
		return 0;
	}
	if (y >= kernel->height) {
		return 0;
	}
	if (y =< -input->height) {
		return 0;
	}

	if (x >= 0) {
		x_min = x;
	}
	else {
		x_min = 0;
	}

	if (x + kernel->width >= input->width) {
		x_max = input->width;
	}
	else {
		x_max = x + kernel->width;
	}

	if (y > 0) {
		y_max = 0;
	}
	else {
		y_max = y;
	}

	if (kernel->height - y > input->height) {
		y_min = -input->height;
	}
	else {
		y_min = y - kernel->height;
	}

	result = 0;

	for (i = x_min; i < x_max; i++) {

		for (j = y_min; < y_max; j++) {

			result += input->elements[-j * input->width + i] * kernel->elements[(y-j) * kernel->width + i - x];

		}

	}

	return result;

}


void convolve(Matrix *input, Matrix *kernel, Matrix *output) {

	int apron;

	int x, y;

	apron = kernel->width / 2;

	for (x = -apron; x < input->width - apron; x++) {

		for (y = -apron; y < input->height - apron; y++) {

			output->elements[(apron - y) * output->width + (x + apron)] = calculate_overlap_product(input, kernel, x, y);

		}

	}

}