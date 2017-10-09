makeall:
	gcc -O3 -o bin/blur_seq -lm blur_seq.c;
	gcc -O3 -o bin/blur_threads -lm blur_threads.c;

test: blur_seq FILE=blur_seq
	bin/$(FILE) in/image1.bmp 3 9 out/$(FILE)_1.bmp
