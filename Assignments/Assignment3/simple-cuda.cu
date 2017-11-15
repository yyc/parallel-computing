__global void hello(char *a, int len){
  int tid = threadIdx.x;
  if(tid > len) return;

  a[tid] += 'A' - 'a';
}
