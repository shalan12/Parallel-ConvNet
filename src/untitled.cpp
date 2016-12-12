/*if (h < xdims[1] && w < xdims[2]) 
      sharedX[tz][ty] = X[getXIdx(n,h,w,c)];
    else 
      sharedX[tz][ty] = 0.0f;


    if (tz + TILE_SIZE  < input_TILE_SIZE && ty + TILE_SIZE < input_TILE_SIZE) {
      if (h + TILE_SIZE < xdims[1] && w + TILE_SIZE < xdims[2]) 
        sharedX[tz + TILE_SIZE][ty + TILE_SIZE] = X[getXIdx(n,h + TILE_SIZE,w + TILE_SIZE,c)];
      else sharedX[tz + TILE_SIZE][ty + TILE_SIZE] = 0.0f;*/

/*if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("Printing sharedX\n");
    for(int ii = 0; ii < input_TILE_SIZE; ii++){
      for(int jj  = 0; jj < input_TILE_SIZE; jj++){
          if (X[getXIdx(n,h+ii,w+jj,c)] != sharedX[ii][jj]) 
          printf("shared[%d][%d] = %f ---- X[%d][%d][%d][%d] = X[%d] = %f", ii, jj, sharedX[ii][jj], n,h+ii,w+jj,c, getXIdx(n,h+ii,w+jj,c) , X[getXIdx(n,h+ii,w+jj,c)]);
      }
      printf("\n");
    }*/
   /* printf("Printing X\n");
    for(int ii = 0; ii < input_TILE_SIZE; ii++){
      for(int jj  = 0; jj < input_TILE_SIZE; jj++){
          printf("%f ", X[getXIdx(n,h+ii,w+jj,c)]);
      }
      printf("\n");
      }*/
  //}