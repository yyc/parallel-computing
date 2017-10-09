buildall: seq_build thread_build

seq_build:
	gcc -O3 -o bin/blur_seq -lm blur_seq.c;

thread_build:
	gcc -O3 -o bin/blur_threads -lm blur_threads.c;


threads: thread_build
	bin/blur_threads in/image1.bmp 3 9 out/image1_threads.bmp
	diff out/image1_ref.bmp out/image1_threads.bmp

seq: seq_build
	bin/blur_seq in/image1.bmp 3 9 out/image1_ref.bmp
