#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>


#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

struct blur_info {
  unsigned char *src;
  float         *dst, sigma;
  int            width, height, ksize;
};
struct write_info {
  unsigned char *write_buf;
  float         *dataB, *dataG, *dataR;
  int            row, numrows, height, width, row_padded;
};


int read_BMP(char           *filename,
             unsigned char  *info,
             unsigned char **data,
             unsigned char **dataR,
             unsigned char **dataG,
             unsigned char **dataB,
             int            *size,
             int            *width,
             int            *height,
             int            *offset,
             int            *row_padded)
{
  int i = 0, j, k, read_bytes, h, w, o, p;

  FILE *f = fopen(filename, "rb");

  if (f == NULL)
  {
    printf("Invalid filename: %s\n", filename);
    return -1;
  }


  read_bytes = fread(info, sizeof(unsigned char), 54, f); // read the 54-byte
                                                          // header


  if (read_bytes != 54)
  {
    printf("Error at read: %d instead of 54 bytes", read_bytes);
    return -1;
  }


  // extract image data from header
  *width      = *(int *)&info[18];
  *height     = *(int *)&info[22];
  *size       = *(int *)&info[2];
  *offset     = *(int *)&info[10];
  *row_padded = (*width * 3 + 3) & (~3);


  // printf ("Filename: %s, Width: %d, Row_padded: %d, Height: %d, Size:  %d,
  // Offset: %d\n", filename, *width, *row_padded, *height, *size, *offset);
  w = *width;
  p = *row_padded;
  h = *height;
  o = *offset;

  *data  = (unsigned char *)malloc(p * h);
  *dataR = (unsigned char *)malloc(w * h);
  *dataG = (unsigned char *)malloc(w * h);
  *dataB = (unsigned char *)malloc(w * h);

  fseek(f, sizeof(unsigned char) * o, SEEK_SET);
  read_bytes = fread(*data, sizeof(unsigned char), p * h, f);


  if (read_bytes != p * h)
  {
    printf("Error at read: %d\n", read_bytes);
    free(data);
    return -1;
  }

  for (k = 0; k < h; k++)
  {
    i = k * p;

    for (j = 0; j < w; j++)
    {
      (*dataB)[k * w + j] = (*data)[i];
      (*dataG)[k * w + j] = (*data)[i + 1];
      (*dataR)[k * w + j] = (*data)[i + 2];

      i += 3;
    }
  }

  fclose(f);
  return 0;
}

void* assemble_segment(void *write_info_ptr, int row, int numrows) {
  struct write_info *info = (struct write_info *)write_info_ptr;
  unsigned char     *write_buf = info->write_buf;
  float *dataB = info->dataB, *dataG = info->dataG, *dataR = info->dataR;
  int    heightlimit = MIN(info->height, row + numrows);
  int    i;

  for (i = row; i < heightlimit; i++) { //
    // Rows
    int write_offset = i * info->row_padded;
    int read_offset  = i * info->width;
    int j;

    for (j = 0; j < info->row_padded / 3; j++) {
      if (j < info->width) {
        write_buf[write_offset + 3 *
                  j] = (unsigned char)dataB[read_offset + j];
        write_buf[write_offset + 3 * j +
                  1] = (unsigned char)dataG[read_offset + j];
        write_buf[write_offset + 3 * j +
                  2] = (unsigned char)dataR[read_offset + j];
      } else {
        write_buf[write_offset + 3 * j]     = (unsigned char)0;
        write_buf[write_offset + 3 * j + 1] = (unsigned char)0;
        write_buf[write_offset + 3 * j + 2] = (unsigned char)0;
      }
    }
  }
  return NULL;
}

int write_BMP(char          *filename,
              unsigned char *write_buf,
              unsigned char *header,
              int            offset,
              int            width,
              int            row_padded,
              int            height)
{
  int   write_bytes = 0, i, pad_size;
  FILE *f = fopen(filename, "wb");

  write_bytes = fwrite(header, sizeof(unsigned char), offset, f);

  if (write_bytes < offset)
  {
    printf("Error at writing the header\n");
    return -1;
  }


  write_bytes = fwrite(write_buf, sizeof(unsigned char), height * row_padded, f);

  if (write_bytes != height * row_padded)
  {
    printf("Error at write: i = %d %d\n", i, write_buf[i]);
    return -1;
  }

  fclose(f);
  return 0;
}

float convolve(const float *kernel, const float *buffer, const int ksize) {
  float sum = 0.0f;
  int   i;

  for (i = 0; i < ksize; i++)
  {
    sum += kernel[i] * buffer[i];
  }
  return sum;
}

void* gaussian_blur(void *blur_info_ptr)
{
  int x, y, i, x1, y1;


  struct blur_info *info_ptr = (struct blur_info *)blur_info_ptr;

  // Deconstruct the original arguments
  unsigned char *src = info_ptr->src;
  float *dst         = info_ptr->dst;
  int    width       = info_ptr->width;
  int    height      = info_ptr->height;
  float  sigma       = info_ptr->sigma;
  int    ksize       = info_ptr->ksize;

  int halfksize = ksize / 2;
  float  sum = 0.f, t;
  float *kernel, *buffer;

  // create Gaussian kernel
  kernel = (float *)malloc(ksize * sizeof(float));
  buffer = (float *)malloc(ksize * sizeof(float));

  if (!kernel || !buffer)
  {
    printf("Error in memory allocation!\n");
    return NULL;
  }

  // if sigma too small, just copy src to dst
  if (ksize <= 1)
  {
    for (y = 0; y < height; y++)
      for (x = 0; x < width; x++) dst[y * width + x] = src[y * width + x];
    return NULL;
  }


  // compute the Gaussian kernel values
  for (i = 0; i < ksize; i++)
  {
    x = i - halfksize;
    t = expf(-x * x / (2.0f * sigma * sigma)) /
        (sqrt(2.0f * M_PI) * sigma);
    kernel[i] = t;
    sum      += t;
  }

  for (i = 0; i < ksize; i++)
  {
    kernel[i] /= sum;

    // printf ("Kernel [%d] = %f\n", i, kernel[i]);
  }


  // blur each row
  // #pragma omp parallel for schedule(static) private(x1, i, buffer)

  for (y = 0; y < height; y++)
  {
    // Set up the buffer, which is a sort of sliding window for the kernel
    // everything to the left is the same as the first elem
    // suppose image is 123456789abcdefghihjkl
    for (x1 = 0; x1 < halfksize; x1++)
    {
      buffer[x1] = (float)src[y * width]; // buffer: 1111_____
    }

    // Everything to the right as usual
    for (x1 = halfksize; x1 < ksize - 1; x1++)
    {
      buffer[x1] = (float)src[y * width + x1 - halfksize];

      /* buffer:
         11111____
         111112
         1111123
         11111234_
       */
    }

    // For each element, shift the buffer one to the right and convolve
    for (x1 = 0; x1 < width; x1++)
    {
      i = (x1 + ksize - 1) % ksize; // 8

      if (x1 < width - halfksize)
      {
        buffer[i] = (float)src[y * width + x1 + halfksize];

        // kernel: 123454321
        // buffer: 671112345
      }

      // Handle the end of the row the same as above
      else
      {
        buffer[i] = (float)src[y * width + width - 1];
      }

      dst[y * width + x1] = convolve(kernel, buffer, ksize);
    }
  }

  // blur each column
  // #pragma omp parallel for schedule(static) private(y1, i, buffer)

  for (x = 0; x < width; x++)
  {
    for (y1 = 0; y1 < halfksize; y1++)
    {
      buffer[y1] = dst[0 * width + x];
    }

    for (y1 = halfksize; y1 < ksize - 1; y1++)
    {
      buffer[y1] = dst[(y1 - halfksize) * width + x];
    }

    for (y1 = 0; y1 < height; y1++)
    {
      i = (y1 + ksize - 1) % ksize;

      if (y1 < height - halfksize)
      {
        buffer[i] = dst[(y1 + halfksize) * width + x];
      }
      else
      {
        buffer[i] = dst[(height - 1) * width + x];
      }

      dst[y1 * width + x] = convolve(kernel, buffer, ksize);
    }
  }


  // clean up
  free(kernel);
  free(buffer);
  return NULL;
}

long long wall_clock_time()
{
#ifdef __linux__
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);

#else /* ifdef __linux__ */
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);

#endif /* ifdef __linux__ */
}

int main(int argc, char **argv)
{
  unsigned char info[54], *dataR = NULL, *dataG = NULL, *dataB = NULL,
                *data = NULL;
  int blur_size, ret_code = 0, size, width, height, offset, row_padded;
  char  *in_filename, *out_filename;
  float *dstB, *dstR, *dstG, sigma;

  long long time_difference, start_time;

  if (argc != 5)
  {
    printf("Usage: %s <filename.bmp> <sigma> <blur_size> <output_filename.bmp>",
           argv[0]);
    return -1;
  }
  in_filename  = argv[1];
  out_filename = argv[4];
  blur_size    = atoi(argv[3]);
  sigma        = atof(argv[2]);


  start_time = wall_clock_time();
  ret_code   = read_BMP(in_filename,
                        info,
                        &data,
                        &dataR,
                        &dataG,
                        &dataB,
                        &size,
                        &width,
                        &height,
                        &offset,
                        &row_padded);

  // printf("Read_BMP took %1.2f seconds\n",
  //        ((float)(wall_clock_time() - start_time)) / 1000000000);


  if (ret_code < 0)
  {
    free(dataB);
    free(dataR);
    free(dataG);
    return -1;
  }

  start_time = wall_clock_time();

  dstB = (float *)malloc(width * height * sizeof(float));
  dstR = (float *)malloc(width * height * sizeof(float));
  dstG = (float *)malloc(width * height * sizeof(float));

  // printf("malloc took %1.2f seconds\n",
  //        ((float)(wall_clock_time() - start_time)) / 1000000000);


  struct blur_info blue_info;
  blue_info.src    = dataB;
  blue_info.dst    = dstB;
  blue_info.width  = width;
  blue_info.height = height;
  blue_info.sigma  = sigma;
  blue_info.ksize  = blur_size;
  struct blur_info red_info;
  red_info.src    = dataR;
  red_info.dst    = dstR;
  red_info.width  = width;
  red_info.height = height;
  red_info.sigma  = sigma;
  red_info.ksize  = blur_size;
  struct blur_info green_info;
  green_info.src    = dataG;
  green_info.dst    = dstG;
  green_info.width  = width;
  green_info.height = height;
  green_info.sigma  = sigma;
  green_info.ksize  = blur_size;


  omp_set_num_threads(64);
  omp_set_nested(1);

  int thread_id;
  start_time = wall_clock_time();
  #pragma omp parallel private(thread_id) num_threads(3)
  {
    thread_id = omp_get_thread_num();

    switch (thread_id) {
    case 0:
      gaussian_blur(&green_info);
      break;

    case 1:
      gaussian_blur(&blue_info);
      break;

    case 2:
      gaussian_blur(&red_info);
      break;
    }
  }


  // printf("Parallel gaussian took %1.2f seconds\n",
  //        ((float)(wall_clock_time() - start_time)) / 1000000000);


  int increment = 1 + height / omp_get_num_procs();
  omp_set_dynamic(0);
  omp_set_nested(0);
  start_time = wall_clock_time();

  int i;
  #pragma omp parallel for private(thread_id)


  for (i = 0; i < omp_get_num_procs(); i++) {
    struct write_info *info = (struct write_info *)malloc(
      sizeof(struct write_info));

    info->write_buf  = data;
    info->dataB      = dstB;
    info->dataR      = dstR;
    info->dataG      = dstG;
    info->row_padded = row_padded;
    info->height     = height;
    info->width      = width;

    assemble_segment(info, i * increment, increment);
  }

  // printf("Assembly took %1.2f seconds\n",
  //        ((float)(wall_clock_time() - start_time)) / 1000000000);

  start_time = wall_clock_time();
  ret_code   = write_BMP(out_filename,
                         data,
                         info,
                         offset,
                         width,
                         row_padded,
                         height);

  // printf("write_BMP took %1.2f seconds\n",
  //        ((float)(wall_clock_time() - start_time)) / 1000000000);

  start_time = wall_clock_time();

  free(dstB);
  free(dstR);
  free(dstG);
  free(dataB);
  free(dataR);
  free(dataG);

  // printf("free took %1.2f seconds\n",
  //        ((float)(wall_clock_time() - start_time)) / 1000000000);


  return ret_code;
}
