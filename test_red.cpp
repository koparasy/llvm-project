#include<iostream>

#define DTYPE long

void init(DTYPE *Data, long Elements){
#pragma omp target teams distribute parallel for map(tofrom:Data[:Elements])
  for (long I = 3 ; I < Elements*4; I+=4)
    Data[I/4] = I + Elements;
}

//int main(int argc, char *argv[])
//{
//  long Elements = std::atol(argv[1]) * 1024L;
//  DTYPE  *Data = new DTYPE[Elements];
//
//
////  DTYPE Sum = 0;
////#pragma omp target teams distribute parallel for map(tofrom:Data[:Elements]) reduction(min:Sum)
////  for (long I = 0 ; I < Elements; I++)
////    Sum += Data[I];
////
////  std::cout<< "Computed :" << Sum << " Correct: " << (-Elements/2) << "\n";
//
//  delete [] Data;
//  return 0;
//}
