#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<math.h>
#include<cblas.h>

#define MAX_FILENAME_SIZE 128

static void getParams(char* parameterFile, int* numRows, int* numSigCols, int* numTotalCols, 
		      int* sigColsOffset, float* intercept, float* noiseLevel, 
		      char* solnfilename, char* bfilename, char* Afilename, char* modelType);
static float generateStandardNormal();
static float generateCoinFlip(float probOfZero);

int main(int argc, char **argv)
{
  //SEED RANDOM NUMBER
  srand(time(NULL));
  //fprintf(stdout, "NORMAL SAMPLES: %f, %f, %f \n", 
  //	  generateStandardNormal(), generateStandardNormal(), generateStandardNormal());

  //VARIABLE DECLARATIONS
  int numRows, numSigCols, numTotalCols, sigColsOffset, i, j;
  float intercept, noiseLevel, objFunc;
  char solnfilename[MAX_FILENAME_SIZE];
  char bfilename[MAX_FILENAME_SIZE];
  char Afilename[MAX_FILENAME_SIZE];
  char modelType;

  fprintf(stdout, "fetching dataGenerator parameters...\n");
  //GET VALUES FROM PARAMETER FILE
  getParams(argv[1], &numRows, &numSigCols, &numTotalCols, &sigColsOffset,
	    &intercept, &noiseLevel, solnfilename, bfilename, Afilename, &modelType);

  fprintf(stdout, "generating data matrix and observations...\n");
  //GENERATE SIGNIFICANT PART OF A MATRIX
  float* A = (float*)malloc(numRows*numSigCols*sizeof(float));
  if(!A) {
    fprintf(stderr,"Error 1.1 - malloc failed\n");
    exit(EXIT_FAILURE);
  }
  for(i=0; i<numRows; i++) {
    for(j=0; j<numSigCols; j++) {
      A[i*numSigCols + j] = generateStandardNormal();
    }
  }

  //GENERATE NONZERO PART OF SOLUTION
  float* soln = (float*)malloc(numSigCols*sizeof(float));
  if(!soln) {
    fprintf(stderr,"Error 1.2 - malloc failed\n");
    exit(EXIT_FAILURE);
  }
  float multiplier;
  if(modelType == 'o')
    multiplier = 3.0;
  else
    multiplier = 1.0;
  for(j=0; j<numSigCols; j++)
    soln[j] = multiplier * generateStandardNormal();

  //GENERATE OBSERVATION VECTOR
  float* b = (float*)malloc(numRows*sizeof(float));
  if(!b) {
    fprintf(stderr,"Error 1.3 - malloc failed\n");
    exit(EXIT_FAILURE);
  }
  cblas_sgemv(CblasRowMajor, CblasNoTrans, numRows, numSigCols, 1.0, A, numSigCols,
	      soln, 1, 0.0, b, 1);
  for(i=0; i<numRows; i++)
    b[i] += intercept + noiseLevel * generateStandardNormal();
  if(modelType == 'o') {
    //b so far represents logodds; need to convert it
    for(i=0; i<numRows; i++)
      b[i] = generateCoinFlip( 1.0 / (1.0 + exp(b[i]) ) );
  }

  //CALCULATE OBJECTIVE FUNCTION VALUE
  objFunc = cblas_sasum(numSigCols, soln, 1);
  float* calcVector = (float*)malloc(numRows*sizeof(float));
  if(!calcVector) {
    fprintf(stderr,"Error 1.4 - malloc failed\n");
    exit(EXIT_FAILURE);
  }
  for(i=0; i<numRows; i++) //set calcVector to intercept value
    calcVector[i] = intercept;
  cblas_sgemv(CblasRowMajor, CblasNoTrans, numRows, numSigCols, 1.0, A, numSigCols,
	      soln, 1, 1.0, calcVector, 1);
  if(modelType == 'l') {
    cblas_saxpy(numRows, -1.0, b, 1, calcVector, 1); //calcVector now holds A*soln - b
    objFunc += pow( cblas_snrm2(numRows, calcVector, 1 ), 2);
  }
  else {
    for(i=0; i < numRows; i++) 
      objFunc += log( 1 + exp( calcVector[i] )) - calcVector[i] * b[i];
  }

  fprintf(stdout, "writing data to files...\n");
  //WRITE SOLN VECTOR TO FILE
  FILE* solnFile = fopen(solnfilename, "w");
  if(solnFile==NULL) {
    fprintf(stderr,"Error 2.1 - File open failed\n");
    exit(EXIT_FAILURE);
  }
  fprintf(solnFile, "Objective Func Value for this vector: %f \n", objFunc);
  fprintf(solnFile, "\nIntercept: %f \n\nSolutionVector:\n", intercept);
  for(j=0; j<numTotalCols; j++) {
    if(j<sigColsOffset || j >= sigColsOffset+numSigCols)
      fprintf(solnFile, "%f \n", 0.0);
    else
      fprintf(solnFile, "%f \n", soln[j - sigColsOffset]);
  }
  fclose(solnFile);

  //WRITE OBSERVATION VECTOR TO FILE
  FILE* vecFile = fopen(bfilename, "w");
  if(vecFile==NULL) {
    fprintf(stderr,"Error 2.2 - File open failed\n");
    exit(EXIT_FAILURE);
  }
  for(i=0; i<numRows; i++) {
    fprintf(vecFile, "%f \n", b[i]);
  }
  fclose(vecFile);

  //WRITE MATRIX TO FILE
  FILE* matFile = fopen(Afilename, "w");
  if(matFile==NULL) {
    fprintf(stderr,"Error 2.3 - File open failed\n");
    exit(EXIT_FAILURE);
  }
  for(i=0; i<numRows; i++) {
    for(j=0; j<numTotalCols; j++) {
      if(j<sigColsOffset || j >= sigColsOffset+numSigCols)
	fprintf(matFile, "%f, ", generateStandardNormal());
      else
	fprintf(matFile, "%f, ", A[i*numSigCols + j - sigColsOffset]);
    }
    fprintf(matFile, "\n");
  }
  fclose(matFile);

  fprintf(stdout,"\nCreated data files %s, %s, %s\n\n", solnfilename, bfilename, Afilename);

  free(A); free(soln); free(b); free(calcVector);
  return 0;
}

static void getParams(char* parameterFile, int* numRows, int* numSigCols, int* numTotalCols, 
		      int* sigColsOffset, float* intercept, float* noiseLevel, 
		      char* solnfilename, char* bfilename, char* Afilename, char* modelType) {
  FILE *paramFile;
  paramFile = fopen(parameterFile, "r");
  if(paramFile == NULL) {
    fprintf(stderr, "Error 0.1 - ParamFile Open Failed!\n");
    exit(EXIT_FAILURE);
  }

  //Read parameters:
  fscanf(paramFile, "FileNameForSoln : %127s", solnfilename);
  fscanf(paramFile, " ModelType : %4s %*128[^\n]", modelType);
  fscanf(paramFile, " numRows : %d", numRows);
  fscanf(paramFile, " numTotalCols : %d", numTotalCols);
  fscanf(paramFile, " numSigCols : %d", numSigCols);
  fscanf(paramFile, " sigColsOffset : %d", sigColsOffset);
  fscanf(paramFile, " intercept : %16f", intercept);
  fscanf(paramFile, " noiseLevel : %16f", noiseLevel);
  fscanf(paramFile, " FileNameForB : %127s", bfilename);
  fscanf(paramFile, " FileNameForA : %127s", Afilename);

  fclose(paramFile);
  return;
}




static float generateStandardNormal() {
  //APPROXIMATE BY ADDING 12 UNIFORM SAMPLES AND SUBTRACTING 6
  float value = 0.0;
  int i;
  for(i=0; i<12; i++) 
    value += (float)rand()/(float)RAND_MAX;
  return value - 6.0;
}

static float generateCoinFlip(float probOfZero) {
  int randomValue = rand() % 1000;
  if( randomValue > (probOfZero * 1000) )
    return 1.0;
  else
    return 0.0;
}
