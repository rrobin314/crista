#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<math.h>
#include<cblas.h>
#include"mpi.h"
#include"CVcristaLib.h"

#define TAG_AX 267
#define TAG_ATX 832
#define TAG_DIE 451
#define TAG_ATAX 674
#define MAX_FILENAME_SIZE 64

static void master(int nslaves, char* parameterFile);
static void slave(int myrank, char* parameterFile);
static void getSlaveParams(char* parameterFile, int* ldA, int* rdA, int* interceptFlag, 
			   char* matrixfilename);
static void getMasterParams(char* parameterFile, char* xfilename, char* bfilename, char* outfilename, 
			    int* rdA, 
			    int* numLambdas, float* lambdaStart, float* lambdaFinish, 
			    int* numFolds, float* gamma, float* step, char* regType, int* accel, 
			    int* MAX_ITER, float* MIN_FUNCDIFF);
static int getVector(float* b, int lengthb, char* bfilename);
static void writeResults(ISTAinstance_mpi* instance, char* outfilename, 
			 char* bfilename, float* lambdas, int numLambdas, float* meanTotalErrors);


int main(int argc, char **argv)
{
  int myrank, nslaves;
  double t1, t2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  t1 = MPI_Wtime();

  //Get number of slaves by getting total number and subtracting 1
  MPI_Comm_size(MPI_COMM_WORLD, &nslaves);
  nslaves = nslaves - 1;

  if (myrank==0)
    {
      master(nslaves, argv[1]);
    }
  else
    {
      slave(myrank, argv[2]);
    }

  t2 = MPI_Wtime();
  if(myrank==0)
    fprintf(stdout, "Elapsed time is %f \n", t2-t1);

  MPI_Finalize();
  return 0;
}

static void master(int nslaves, char* parameterFile)
{
  //VARIABLE DECLARATIONS
  time_t startTime, computationStartTime, endTime;
  int rank, i, j, accel, MAX_ITER, *slave_ldAs, total_ldA, rdA, numLambdas, numFolds, error;
  ISTAinstance_mpi* instance;
  float *xvalue, *xinit, *result, *b, *lambdas, lambdaStart, lambdaFinish, *meanTotalErrors, gamma, step, MIN_FUNCDIFF;
  char regType, xfilename[MAX_FILENAME_SIZE], bfilename[MAX_FILENAME_SIZE], outfilename[MAX_FILENAME_SIZE];

  //START TIMER
  startTime = time(NULL);
  srand(time(NULL));

  //GET VALUES FROM PARAMETER FILE
  getMasterParams(parameterFile, xfilename, bfilename, outfilename, &rdA, 
		  &numLambdas, &lambdaStart, &lambdaFinish, &numFolds, &gamma, &step, &regType, &accel, 
		  &MAX_ITER, &MIN_FUNCDIFF);

  //STORE EACH SLAVE'S INDIVIDUAL LDA AND CALCULATE TOTAL_LDA
  slave_ldAs = (int*)malloc((nslaves+1)*sizeof(int));
  int my_ldA = 0;
  MPI_Gather(&my_ldA, 1, MPI_INT, slave_ldAs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  total_ldA = 0;
  for(i=0; i<=nslaves; i++)
    total_ldA += slave_ldAs[i];
  //fprintf(stdout, "TOTAL LDA IS %d\n", total_ldA);


  //ALLOCATE MEMORY
  xvalue = calloc(rdA+1,sizeof(float));
  result = malloc((total_ldA+rdA)*sizeof(float));
  b      = malloc((total_ldA)*sizeof(float));
  lambdas = malloc(numLambdas*sizeof(float));
  meanTotalErrors = calloc(numLambdas, sizeof(float));
  if(xvalue==NULL || result==NULL || b==NULL || lambdas==NULL || meanTotalErrors==NULL)
    fprintf(stdout,"1.1-Unable to allocate memory!");
  

  //ASSIGN VALUES TO XVALUE AND B
  error=1;
  if(strcmp(xfilename, "zeros")==0){
    //do nothing - calloc already initialized xvalue to 0
  }
  else 
    error *= getVector(xvalue, rdA, xfilename);
  error *= getVector(b, total_ldA, bfilename);

  //CREATE 'xinit' VECTOR TO HOLD THE ORIGINAL 'xvalue' VECTOR
  xinit = malloc((rdA+1)*sizeof(float));
  if(xinit==NULL)
    fprintf(stdout,"1.2-Unable to allocate memory!");
  cblas_scopy(rdA + 1, xvalue, 1, xinit, 1);

  //CHECK FOR FILEOPEN ERRORS; IF ANY PRESENT END PROGRAM
  for(i=1; i<=nslaves; i++)
    if(slave_ldAs[i] == -1) error=0;
  MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(error==0) {
    free(result);
    free(xvalue);
    free(b);
    free(lambdas);
    return;
  }
  
  //PRINT INPUTS
  /*  fprintf(stdout, "Here's x:\n");
  for(i=0; i < rdA; i++)
    {
      fprintf(stdout, "%f ", xvalue[i]);
    }
  fprintf(stdout, "\n and here's b:\n");
  for(i=0; i < total_ldA; i++)
    {
      fprintf(stdout, "%f ", b[i]);
    }
  */

  //CREATE ISTA OBJECT
  instance = ISTAinstance_mpi_new(slave_ldAs, total_ldA, rdA, numFolds, b, lambdaStart, gamma, 
				  accel, regType, xvalue, step,
				  nslaves, MPI_COMM_WORLD,
				  TAG_AX, TAG_ATX, TAG_ATAX, TAG_DIE);
  
  //CENTER FEATURES
  float* shifts = calloc(rdA, sizeof(float));
  MPI_Reduce(shifts, instance->meanShifts, rdA, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  cblas_sscal(rdA, 1.0 / total_ldA, instance->meanShifts, 1); 
  
  MPI_Bcast(instance->meanShifts, rdA, MPI_FLOAT, 0, MPI_COMM_WORLD);

  //SCALE FEATURES
  float* norms = calloc(rdA, sizeof(float));
  MPI_Reduce(norms, instance->scalingFactors, rdA, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  for(j=0; j<rdA; j++)
    instance->scalingFactors[j] = pow(instance->scalingFactors[j], 0.5);

  MPI_Bcast(instance->scalingFactors, rdA, MPI_FLOAT, 0, MPI_COMM_WORLD); 

  //CREATE LAMBDA PATH
  calcLambdas(lambdas, numLambdas, lambdaStart, lambdaFinish, instance);

  //CALCULATE CROSS VALIDATION FOLDS
  calcFolds(instance);

  //DEBUGGING AREA
  /*float* ones = calloc(rdA+1, sizeof(float));
  for(i=0; i< rdA+1; i++) {
    ones[i] = 1.0;
  }
  fprintf(stdout, "meanshifts: ");
  for(i=0; i<5; i++) {
    fprintf(stdout, "%f ", instance->meanShifts[i]);
  }
  fprintf(stdout, "\nscalingFactors: ");
  for(i=0; i<5; i++) {
    fprintf(stdout, "%f ", instance->scalingFactors[i]);
  }
  fprintf(stdout, "\nlambdas: ");
  for(i=0; i<5; i++) {
    fprintf(stdout, "%f ", lambdas[i]);
  }
  fprintf(stdout, "\nA * ones: ");
  multiply_Ax(ones, result, instance);
  for(i=0; i<5; i++) {
    fprintf(stdout, "%f ", result[i]);
  }
  fprintf(stdout, "\n");
  */
  
  //TIME UPDATE
  computationStartTime = time(NULL);

  //LOOP THROUGH EACH FOLD
  fprintf(stdout, "Begin ISTA calc...\n");
  for(i=0; i < instance->numFolds; i++) {
    instance->currentFold = i;

    //RESET XVALUE TO BE XINIT AND STEPSIZE TO INTIAL VALUE
    cblas_scopy(rdA + 1, xinit, 1, instance->xcurrent, 1);
    *(instance->stepsize) = step;

    //LOOP THROUGH EACH LAMBDA
    for(j=0; j < numLambdas; j++) {
      instance->lambda = lambdas[j];
      //RUN ISTA ON THIS PARTICULAR LAMBDA AND FOLD
      ISTAsolve_liteCV(instance, MAX_ITER, MIN_FUNCDIFF);
      fprintf(stdout, "fold %d and lambda %f - trainingerror: %f, testerror: %f \n",
	      i, lambdas[j], 
	      ISTAloss_func_mpiCV(instance->xcurrent, instance, 0) * instance->numFolds / (float) (instance->numFolds - 1) / (float) instance->ldA,
	      ISTAloss_func_mpiCV(instance->xcurrent, instance, 1) * instance->numFolds / (float) instance->ldA);

      //ADD TEST ERROR TO 'meanTotalErrors' VECTOR
      meanTotalErrors[j] += ISTAloss_func_mpiCV(instance->xcurrent, instance, 1) / (float)instance->ldA;
    }
  }

  //UNDO RESCALING
  //for(i=0; i<(instance->rdA); i++) {
  //  if(instance->scalingFactors[i] > 0.0001)
  //   instance->xcurrent[i] = instance->xcurrent[i] / instance->scalingFactors[i];
  //}
  //instance->intercept = -1.0 * cblas_sdot(instance->rdA, instance->xcurrent, 1, instance->meanShifts, 1);

  //WRITE RESULTS
  writeResults(instance, outfilename, bfilename, lambdas, numLambdas, meanTotalErrors);

  //STOP TIME
  endTime = time(NULL);
  fprintf(stdout,"Setup took %f seconds and computation took %f seconds\n",
	  difftime(computationStartTime, startTime), difftime(endTime, computationStartTime));

  //CLOSE THE SLAVE PROCESSES AND FREE MEMORY
  fprintf(stdout, "Closing the program\n");
  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG_DIE, MPI_COMM_WORLD);
    }

  free(result); 
  ISTAinstance_mpi_free(instance); 
  free(xinit);
  free(meanTotalErrors);
  free(shifts); 
  free(norms); 
  free(lambdas);
  return;
}




static void slave(int myrank, char* parameterFile)
{
  int i, j, dummyInt, target_ldA, my_ldA, rdA, interceptFlag, error;
  MPI_Status status;
  float *A, *xvalue, *resultVector, *tempHolder, *dummyFloat;
  dummyFloat=0;
  char matrixfilename[MAX_FILENAME_SIZE];

  //GET PARAMETERS FROM THE TEXT FILE
  getSlaveParams(parameterFile, &target_ldA, &rdA, &interceptFlag, matrixfilename);

  //ALLOCATE A, TEMPHOLDER, RESULTVECTOR and XVALUE
  A = malloc(target_ldA*(rdA+1)*sizeof(float));
  if(A==NULL)
    fprintf(stdout,"3.1-Unable to allocate memory!");

  xvalue = malloc( (target_ldA+rdA)*sizeof(float) );
  tempHolder = malloc( (target_ldA+rdA)*sizeof(float) ); //place holder for intermediate calculations
  resultVector = malloc( (target_ldA+rdA)*sizeof(float) );
  if(xvalue==NULL || tempHolder==NULL || resultVector==NULL)
    fprintf(stdout,"3.2-Unable to allocate memory!");


  //FILL A WITH DESIRED VALUES AND SEND NUMBER OF FILLED ROWS TO MASTER
  my_ldA=get_dat_matrix(A, target_ldA, rdA, myrank, matrixfilename, interceptFlag);
  MPI_Gather(&my_ldA, 1, MPI_INT, &dummyInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(error==0) { //if there were file open errors, end program
    free(A);
    free(xvalue);
    free(tempHolder);
    free(resultVector);
    return;
  }
  fprintf(stdout,"Slave %d found %d valid rows: A[0] is %f \n", myrank, my_ldA, A[0] );
  

  //CENTER FEATURES
  float* shifts = malloc((rdA+1)*sizeof(float));
  float* ones = malloc(my_ldA*sizeof(float));
  for(i=0; i<my_ldA; i++)
    ones[i] = 1.0;
  cblas_sgemv(CblasRowMajor, CblasTrans, my_ldA, rdA+1, 1.0, A, rdA+1, 
	      ones, 1, 0.0, shifts, 1); //shifts now holds the sums of the columns of A
  MPI_Reduce(shifts, dummyFloat, rdA, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
 
  MPI_Bcast(shifts, rdA, MPI_FLOAT, 0, MPI_COMM_WORLD); //shifts now holds the total means of the columns of A
  for(i=0; i<my_ldA; i++) { //Now we substract shifts from each row of A
    cblas_saxpy(rdA, -1.0, shifts, 1, &A[i*(rdA+1)], 1);
  }

  //SCALE FEATURES
  float* norms = calloc(rdA, sizeof(float));
  for(i=0; i<my_ldA; i++) {
    for(j=0; j<rdA; j++) {
	norms[j] += pow( A[i*(rdA+1) + j], 2);
    }
  }
  MPI_Reduce(norms, dummyFloat, rdA, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(norms, rdA, MPI_FLOAT, 0, MPI_COMM_WORLD); //norms now holds the 2-norms of the total columns of A
  for(j=0; j<rdA; j++) {
    if(norms[j] > 0.0001)
      cblas_sscal(my_ldA, 1.0 / norms[j], A + j, rdA + 1);
  }


  //COMPUTATION LOOP
  while(1)
    {
      MPI_Recv(&dummyInt, 0, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      //Check the tag to determine what to do next
      if (status.MPI_TAG == TAG_DIE)
	{
	  break;
	}

      else if (status.MPI_TAG == TAG_AX)
	{
	  //Multiply A * x

	  //Get xvalue
	  MPI_Bcast(xvalue, rdA+1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	  //Multiply: resultVector = A*xvalue
	  cblas_sgemv(CblasRowMajor, CblasNoTrans, my_ldA, rdA+1, 1.0, A, rdA+1, 
		      xvalue, 1, 0.0, resultVector, 1);

	  //Gather xvalues
	  MPI_Gatherv(resultVector, my_ldA, MPI_FLOAT, dummyFloat, &dummyInt, &dummyInt, MPI_FLOAT, 0, MPI_COMM_WORLD);

	  
	}

      else if (status.MPI_TAG == TAG_ATX)
	{
	  //Multiply A^t * x
	  
	  //Get xvalue
	  MPI_Scatterv(dummyFloat, &dummyInt, &dummyInt, MPI_FLOAT, 
		       xvalue, my_ldA, MPI_FLOAT, 0, MPI_COMM_WORLD);

	  //Multiply: resultVector = A'*xvalue
	  cblas_sgemv(CblasRowMajor, CblasTrans, my_ldA, rdA+1, 1.0, A, rdA+1, 
		      xvalue, 1, 0.0, resultVector, 1);

	  //Sum resultVectors to get final result
	  MPI_Reduce(resultVector, dummyFloat, rdA+1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


	}
      else if (status.MPI_TAG == TAG_ATAX)
	{
	  //Multiply A^t * A * x

	  //Get xvalue
	  MPI_Bcast(xvalue, rdA+1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	  //Multiply: tempHolder = A*xvalue
	  cblas_sgemv(CblasRowMajor, CblasNoTrans, my_ldA, rdA+1, 1.0, A, rdA+1, 
		      xvalue, 1, 0.0, tempHolder, 1);
	  //Multiply: resultVector = A^t * tempHolder
	  cblas_sgemv(CblasRowMajor, CblasTrans, my_ldA, rdA+1, 1.0, A, rdA+1,
		      tempHolder, 1, 0.0, resultVector, 1);

	  //Gather and sum results
	  MPI_Reduce(resultVector, dummyFloat, rdA+1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	  
	}

    }

  free(A); free(xvalue); free(tempHolder); free(resultVector); free(shifts); free(ones); free(norms);
  return;
}


static void getMasterParams(char* parameterFile, char* xfilename, char* bfilename, char* outfilename, 
			    int* rdA, 
			    int* numLambdas, float* lambdaStart, float* lambdaFinish, 
			    int* numFolds, float* gamma, float* step, char* regType, int* accel, 
			    int* MAX_ITER, float* MIN_FUNCDIFF) {
  FILE *paramFile;
  paramFile = fopen(parameterFile, "r");
  if(paramFile == NULL)
    fprintf(stderr, "ParamFile Open Failed!\n");

  //Read parameters:
  fscanf(paramFile, "FileNameForX0 : %63s", xfilename);
  fscanf(paramFile, " FileNameForB : %63s", bfilename);
  fscanf(paramFile, " OutputFile : %63s", outfilename);
  fscanf(paramFile, " numCols : %d", rdA);
  fscanf(paramFile, " numLambdas : %d %*128[^\n]", numLambdas);
  fscanf(paramFile, " lambdaStart : %16f %*128[^\n]", lambdaStart);
  fscanf(paramFile, " lambdaFinish : %16f", lambdaFinish);
  fscanf(paramFile, " numFolds : %d", numFolds);
  fscanf(paramFile, " StepSizeDecretion : %16f", gamma);
  fscanf(paramFile, " InitialStep : %16f", step);
  fscanf(paramFile, " RegressionType : %4s", regType);
  fscanf(paramFile, " FistaAcceleration : %d", accel);
  fscanf(paramFile, " MaximumIterations : %d", MAX_ITER);
  fscanf(paramFile, " MinimumFuncDelta : %16f", MIN_FUNCDIFF);

  fclose(paramFile);
  return;
}


static void getSlaveParams(char* parameterFile, int* ldA, int* rdA, int* interceptFlag, 
			   char* matrixfilename) {

  FILE *paramFile;
  paramFile = fopen(parameterFile, "r");
  if(paramFile == NULL)
    fprintf(stderr, "ParamFile Open Failed!\n");

  fscanf(paramFile, "MatrixFileName : %63s", matrixfilename);
  fscanf(paramFile, " numRows : %d", ldA);
  fscanf(paramFile, " numCols : %d", rdA);
  fscanf(paramFile, " interceptFlag : %d", interceptFlag);

  fclose(paramFile);
  return;
}

static int getVector(float* b, int lengthb, char* bfilename) {
  FILE *paramFile;
  paramFile = fopen(bfilename, "r");
  if(paramFile == NULL) {
    fprintf(stderr, "File open failed for %s! Exiting program...\n", bfilename);
    return 0;
  }

  int i;
  float value;
  for(i=0; i<lengthb; i++) {
    fscanf(paramFile, " %32f , ", &value);
    b[i] = value;
  }

  fclose(paramFile);
  return 1;
}

static void writeResults(ISTAinstance_mpi* instance, char* outfilename, 
			 char* bfilename, float* lambdas, int numLambdas, float* meanTotalErrors) {
  char regForm[10] = "linear";
  char accelForm[10] = "FISTA";
  int i;

  if(instance->regressionType == 'o')
    strcpy(regForm, "logistic");
  if(instance->acceleration == 0)
    strcpy(accelForm, "ISTA");
  
  FILE* outFILE;
  outFILE = fopen(outfilename,"w");
  if(outFILE!=NULL) {
    fprintf(outFILE, "Cross validation results for %s regression using %s algorithm. \n", regForm, accelForm);
    fprintf(outFILE, "Using data from:\nVector File %s \n", bfilename);
    fprintf(outFILE, "Cross validation carried out with %d random folds \n", instance->numFolds);
    fprintf(outFILE, "First column shows lambdas and second column shows average test error\n");
    for(i=0; i < numLambdas; i++) {
      fprintf(outFILE, "%f  %f\n", lambdas[i], meanTotalErrors[i]);
    }
    fclose(outFILE);
  }
}

