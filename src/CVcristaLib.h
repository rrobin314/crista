typedef struct ISTAinstance_mpi {
  // PARAMETERS
  // NOTE: A is now stored with the slaves, only the dimensions are here
  int ldA; //left dimension of A
  int* slave_ldAs; //vector containing ldA for each slave's piece of A
  int* slave_ldAs_displacements; //transformation of above vector used in MPI_Gatherv
  int rdA; //right dimension of A
  float* b;
  float lambda; //how much weight we give to the 1-norm
  float gamma; //fraction we decrease stepsize each time it fails
  int acceleration; //0 if normal ISTA update; 1 for FISTA acceleration
  int interceptFlag; //1 to add a column of ones to A to calculate an intercept
  char regressionType; // 'l' for linear regression and 'o' for logistic regression

  // SCALING VALUES
  float* meanShifts;
  float* scalingFactors;
  float intercept;

  // MPI VALUES
  int nslaves;
  MPI_Comm comm;
  int tag_ax;
  int tag_atax;
  int tag_atx;
  int tag_die;

  //CROSS VALIDATION VALUES
  int currentFold;
  int numFolds;
  int* folds;

  // VALUES DURING CALCULATION
  float* stepsize;
  float* xcurrent;
  float* xprevious;
  float* searchPoint; //necessary when using FISTA acceleration
  float* gradvalue;
  float* eta; //intermediate space used during calculations, usually holds A*xcurrent

} ISTAinstance_mpi;

// "Constructor" for ISTAinstance
// Returns a pointer to an ISTAinstance object with the arguments to the function set as 
// the corresponding elements in the ISTAinstance object.
// Allocates appropriate memory for:
// stepsize, xprevious, searchPoint, gradvalue, eta, meanShifts, and scalingFactors.
// Finally, sets searchPoint equal to xvalue.
extern 
ISTAinstance_mpi* ISTAinstance_mpi_new(int* slave_ldAs, int ldA, int rdA, int numFolds, 
				       float* b, float lambda, 
				       float gamma, int acceleration, char regressionType, 
				       float* xvalue, float step,
				       int nslaves, MPI_Comm comm, int ax, int atx, int atax, int die);


// "Deconstructor" for ISTAinstance
// Applies free to all pointers contained in instance, then applies free to instance itself.
extern void ISTAinstance_mpi_free(ISTAinstance_mpi* instance);


// Runs the ISTA algorithm with the parameters stored in the ISTAinstance_mpi* argument.
// Algorithm terminates when the number of iterations exceeds MAX_ITER
// or when the percent change of the objective function is lower than MIN_FUNCDIFF
extern void ISTAsolve_liteCV(ISTAinstance_mpi* instance, int MAX_ITER, float MIN_FUNCDIFF );

// Backtracking routine to determine how big of a gradient step to take during ISTA.
// Does the following updates:
// Updates gradvalue to that of current searchPoint
// Updates xcurrent to the gradient step from searchPoint indicated by stepsize
// Updates eta to xcurrent - searchPoint
// If additional loops are necessary, updates stepsize to gamma*stepsize 
extern void ISTAbacktrackCV(ISTAinstance_mpi* instance);

// Calculates gradient of ISTAloss_func_mpi at searchPoint and stores it in gradvalue.
// Only uses rows of A not corresponding to those in "currentFold"
extern void ISTAgradCV(ISTAinstance_mpi* instance);

// Calculates the appropriate loss function for either linear or logistic regression
// If insideFold == 1, then we calculate only on those rows in the current fold.
// If insideFold == 0, then we calculate on every row outside of current fold.
extern float ISTAloss_func_mpiCV(float* xvalue, ISTAinstance_mpi* instance, int insideFold);

//Implements soft thresholding operation
extern void soft_threshold(float* xvalue, int xlength, float threshold);

// Calculates the folds used during cross validation.
// Randomly generates a permutation of length ldA and uses this to fill
// the vector "folds" with random fold numbers.
extern void calcFolds(ISTAinstance_mpi* instance);

// Calculates a path of lambdas to solve on:
// If lambdaStart > 0, then our starting point is just lambdaStart
// If lambdaStart < 0, then we calculate our starting point to be:
//     0.5 * || A' * b ||_infinity
// 
// If numLambdas == 1, then our path just consists of lambdaFinish
// If numLambdas > 1, then we draw a line between
// the starting point and lambdaFinish
extern void calcLambdas(float* lambdas, int numLambdas, float lambdaStart, 
			float lambdaFinish, ISTAinstance_mpi* instance);

//Routine that uses MPI to calculate the matrix-vector product A*xvalue and stores it in result.
//The length of xvalue must be rdA+1.
//The returned length of result is ldA.
extern void multiply_Ax(float* xvalue, float* result, ISTAinstance_mpi* instance);

//Same as multiply_Ax, but with A'
//The length of xvalue must be ldA.
//The returned length of result is rdA+1.
extern void multiply_ATx(float* xvalue, float* result, ISTAinstance_mpi* instance);

//Routine that uses MPI to calculate the matrix-vector produce A'*A*xvalue and stores it in result
//The length of xvalue must be rdA+1.
//The returned length of result is rdA+1.
extern void multiply_ATAx(float* xvalue, float* result, ISTAinstance_mpi* instance);

//This method gets an ldA x rdA matrix from the csv file "filename"
//and stores an ldA x rdA+1 matrix in A that corresponds to the matrix from the 
//file plus a final column that is either all-zero or all-one, depending on 
//"interceptFlag"
//
//It is assumed that "filename" contains a matrix without row or column labels
//where floats are specified with fewer than 24 significant digits.
//It is assumed that "filename" contains a matrix with exactly rdA columns
//
//A is located by skipping the first (myrank-1)*ldA rows in the file and
//starting input of A from that point.
extern int get_dat_matrix(float* A, int ldA, int rdA, int myrank, 
			   char* filename, int interceptFlag);
