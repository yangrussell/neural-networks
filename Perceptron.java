import java.io.*;
import java.util.*;
/**
 * The Perceptron class implements a feed-forward neural network with a configurable number
 * of input nodes, output nodes, number of hidden layers, and number of nodes in each 
 * hidden layer. The Perceptron object asks for a configuration file at run-time when its
 * main method is run. The configuration file specifies information such as the number
 * of nodes in each layer, the value of lambda (the learning factor), the maximum number
 * of iterations, a file for weights (or 'randomize' if the weights should be generated randomly),
 * an inputs file, an outputs file, and a lower and upper bound on randomly generated weights.
 * A Perceptron object can be trained on input cases via gradient descent, and will stop training
 * if the error drops below a threshold or a maximum number of iterations is reached. The
 * backpropagation algorithm is used when training the perceptron.
 * 
 * Table of contents (list of methods):
 * public Perceptron(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda,
 *                   int maxIterations, double stoppingError, String weightsFile, String trainingCases,
 *                   String outputsFile, double lowerBound, double upperBound)
 * public Perceptron(String filename)
 * private void setInstanceVariables(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda,
 *                                   int maxIterations, double stoppingError, String weightsFile, String inputsFile,
 *                                   String outputsFile, double lowerBound, double upperBound)
 * private double[][] readOutputs(String outputsFile)
 * private static int arrayMax(int[] arr)
 * private void readWeights(String filename)
 * public void setRandomWeights()
 * public double generateRandom()
 * private double[] runNetwork(double[] inputs, boolean raw)
 * private void readInputs(String filename)
 * private static double activationFunction(double x)
 * private static double activationFunctionDerivative(double x)
 * public void updateWeightsBackprop(double[] theoretical, double[] calculated)
 * public void gradientDescent()
 * private double calculateError(double[] theoreticalOutputs, double[] actualOutputs)
 * private double calculateTotalError(double[] errorArr)
 * 
 * @author Russell Yang
 * @version 9/4/2019 (creation date)
 */
public class Perceptron
{
   // instance variables  
   private int[] layerSizes;                            // A 1D array representing the sizes of each of the layers
   private double[][] activations;                      // A 2D array representing the activation states of the network
   private double[][] rawActivations;                   // A 2D array representing the activation states of the network without activation function
   private double[][][] weights;                        // A 3D array representing the weights of the network
   private double[][] omega;                            // A 2D array representing the omega values in backpropagation
   private double[][] psi;                              // A 2D array representing the psi values in backpropagation
   private double lambda;                               // A value of lambda, the learning factor
   private int maxIterations;                           // A maximum number of iterations that the network will train for
   private double[][] theoreticalOutputs;               // A 2D array - each row is an array of outputs for each input case
   private double lowerBound;                           // Lower bound on the random weights
   private double upperBound;                           // Upper bound on the random weights
   private double[][] trainingCases;                    // A 2D array - each row is an input case
   private static final double LAMBDA_MULTIPLIER = 1.0; // A value to multiply lambda by (not in use now as the network is not adaptive)
   private double stoppingError;                        // The network will drop if the total error drops below this threshold

   /**
    * Constructor for the Perceptron class with parameters. Sets instance variables to values based on the parameters, using the
    * helper method setInstanceVariables.
    * 
    * @param inputNodes the number of nodes that the network uses to take in inputs
    * @param hiddenLayerNodes an array where each element is the number of nodes in a hidden layer of the network,
    *        and the length of the array is the number of hidden layers
    * @param outputNodes the number of nodes in the output layer
    * @param lambda a value of lambda, the learning factor
    * @param maxIterations the maximum number of iterations the network will be trained for
    * @param stoppingError a threshold; the network will stop if the total error drops below it
    * @param weightsFile a path to a file of weights or the word "randomize". If weightsFile is "randomize", weights will
    *        be generated according to a specified lower and upper bound. If the weightsFile is to a file of weights,
    *        the weights are whitespace delimited and each new line represents a different value for the connectivity layer index (m).
    *        For example, for a 2-2-1 network, the text file will be structured as follows: 
    *        w000 w001 w010 w011
    *        w100 w110
    * @param inputsFile a filename of the inputs file, where the first line consists of 2 space separated integers.
    *        The first is the number of cases and the second is the number of inputs per case.
    * @param outputsFile a filename where the file contains the theoretical outputs to be read. The
    *        first line consists of 2 space separated integers, the first is the number of array items in
    *        each row that follow, and the second is the number of rows. Each row in the file after
    *        the first line corresponds to a set of inputs. Within each row, the elements are space-separated
    *        and the first element is the first output of the network, the second element is the second
    *        output of the network, etc. For example, for a neural network that is doing multiple outputs
    *        and is supposed to output OR, AND, and XOR in the first, second, and third outputs, the input
    *        cases would be all the different combinations of two boolean inputs: (0,0); (0,1); (1,0); and (1,1).
    *        Thus, taking the first column to be the OR outputs, the second column to be the AND outputs, and
    *        the third column to be the XOR outputs, the outputsFile would look like this:
    *        3 4
    *        0 0 0
    *        1 0 1
    *        1 0 1
    *        1 1 0
    * @param lowerBound a lower bound (inclusive) on the values of the randomly generated initial weights
    * @param upperBound an upper bound (exclusive) on the values of the randomly generated initial weights
    * @param outputNodes the number of output nodes in the network
    */
   public Perceptron(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda,
                     int maxIterations, double stoppingError, String weightsFile, String trainingCases,
                     String outputsFile, double lowerBound, double upperBound)
   {
      // Call the setInstanceVariables method to set the instance variables to the passed values
      setInstanceVariables(inputNodes, hiddenLayerNodes, outputNodes, lambda, maxIterations, stoppingError,
                           weightsFile, trainingCases, outputsFile, lowerBound, upperBound);
   }

   /**
    * Constructor for the Perceptron class that reads from a given configuration file. Sets instance variables to values
    * based on the read values, using the helper method setInstanceVariables.
    * 
    * @param filename the name of the configuration file
    *               
    * Special considerations: this method performs exception catching to catch an NumberFormatException, FileNotFoundException,
    * or ArrayIndexOutOfBoundsException that may be thrown. It will throw a RuntimeException with a relevant message
    * if either of those occurs
    */
   public Perceptron(String filename)
   {
      /*
       * Use a try-catch construct.
       * 
       * It is possible that some contents of the file are not the type they should be (ex: weights cannot be parsed
       * to double). In that case, catch the NumberFormatException and throw a RuntimeException with a relevant message
       * for the user.
       * 
       * It is also possible that the file is misspecified and cannot be read. In that case, catch the
       * FileNotFoundException and throw a RuntimeException with a relevant message for the user.
       * 
       * It is also possible that when the space-separated values are split into an array and the array is read
       * from, the array index will be accessed out of bounds. In that case, catch the ArrayIndexOutOfBoundsException
       * and throw a RuntimeException with a relevant message for the user.
       */
      try
      {
         File myFile = new File(filename); // Create a File object
         Scanner sc = new Scanner(myFile); // Create a Scanner to scan the File object

         String firstLine = sc.nextLine();                 // Get first scanner line
         int numInputNodes = Integer.parseInt(firstLine);  // Get number of input nodes by parsing the first line to an Integer

         String secondLine = sc.nextLine();                // Get second scanner line
         String[] splitSecondLine = secondLine.split(" "); // Split the second scanner line by spaces and save it to an array of Strings

         int[] hiddenLayerNodesArray = new int[splitSecondLine.length]; // New array to store the # of nodes in each hidden layer

         for (int i = 0; i < splitSecondLine.length; i++) // Iterate over the splitSecondLine array
         {
            // Parse the current element as an Integer and put it into the corresponding slot in the hiddenLayerNodes array
            hiddenLayerNodesArray[i] = Integer.parseInt(splitSecondLine[i]);
         }

         String thirdLine = sc.nextLine();                     // Get third scanner line
         int numOutputNodes = Integer.parseInt(thirdLine);     // Get number of output nodes by parsing the third line to an Integer

         String fourthLine = sc.nextLine();                    // Get fourth scanner line
         double lambda = Double.parseDouble(fourthLine);       // Get lambda by parsing the fourth line to a Double

         String fifthLine = sc.nextLine();                     // Get fifth scanner line
         int maxIterations = Integer.parseInt(fifthLine);      // Get the max number of iterations by parsing the fifth line to an Integer

         String sixthLine = sc.nextLine();                     // Get the sixth scanner line
         double stoppingError = Double.parseDouble(sixthLine); // Get the stopping error threshold by parsing the sixth line to a double

         String weightsFile = sc.nextLine();                   // Get the filename (or the word "randomize") of the weights (the sixth line)

         String inputsFile = sc.nextLine();                    // Get the filename of the inputs file (the seventh line)

         String outputsFile = sc.nextLine();                   // Get the filename of the outputsFile (the seventh line)

         String ninthLine = sc.nextLine();                     // Get ninth scanner line
         String[] bounds = ninthLine.split(" ");               // Split the ninth scanner line by spaces and save it to an array of Strings
         double lowerBound = Double.parseDouble(bounds[0]);    // Get lower bound by parsing the first element of bounds to a Double
         double upperBound = Double.parseDouble(bounds[1]);    // Get upper bound by parsing the second element of bounds to a Double

         sc.close(); // close the scanner object

         // Call the setInstanceVariables method to set the instance variables to the values read from the configuration file
         setInstanceVariables(numInputNodes, hiddenLayerNodesArray, numOutputNodes, lambda, maxIterations, stoppingError,
                              weightsFile, inputsFile, outputsFile, lowerBound, upperBound);
      } // try
      catch (NumberFormatException n)
      {
         // Throw RuntimeException if a parsing error occurs
         throw new RuntimeException("Could not parse the file");
      }
      catch (FileNotFoundException f)
      {
         // Throw RuntimeException if the file cannot be found
         throw new RuntimeException("The file could not be found");
      }
      catch (ArrayIndexOutOfBoundsException a)
      {
         // Throw RuntimeException if an array index is out of bounds
         throw new RuntimeException("Array index out of bounds. Please check the space-separated values in the configuration file");
      }
   }

   /**
    * A helper method that sets instance variable values based on values passed from either of the constructors.
    * 
    * @param inputNodes the number of nodes that the network uses to take in inputs
    * @param hiddenLayerNodes an array where each element is the number of nodes in a hidden layer of the network,
    *        and the length of the array is the number of hidden layers
    * @param outputNodes the number of nodes in the output layer
    * @param lambda a value of lambda, the learning factor
    * @param maxIterations the maximum number of iterations the network will be trained for
    * @param stoppingError a threshold; the network will stop if the total error drops below it
    * @param weightsFile a path to a file of weights or the word "randomize". If weightsFile is "randomize", weights will
    *        be generated according to a specified lower and upper bound. If the weightsFile is to a file of weights,
    *        the weights are whitespace delimited and each new line represents a different value for the connectivity layer index (m).
    *        For example, for a 2-2-1 network, the text file will be structured as follows: 
    *        w000 w001 w010 w011
    *        w100 w110
    * @param inputsFile a filename of the inputs file, where the first line consists of 2 space separated integers.
    *        The first is the number of cases and the second is the number of inputs per case.
    * @param outputsFile a filename where the file contains the theoretical outputs to be read. The
    *        first line consists of 2 space separated integers, the first is the number of array items in
    *        each row that follow, and the second is the number of rows. Each row in the file after
    *        the first line corresponds to a set of inputs. Within each row, the elements are space-separated
    *        and the first element is the first output of the network, the second element is the second
    *        output of the network, etc. For example, for a neural network that is doing multiple outputs
    *        and is supposed to output OR, AND, and XOR in the first, second, and third outputs, the input
    *        cases would be all the different combinations of two boolean inputs: (0,0); (0,1); (1,0); and (1,1).
    *        Thus, taking the first column to be the OR outputs, the second column to be the AND outputs, and
    *        the third column to be the XOR outputs, the outputsFile would look like this:
    *        3 4
    *        0 0 0
    *        1 0 1
    *        1 0 1
    *        1 1 0
    * @param lowerBound a lower bound (inclusive) on the values of the randomly generated initial weights
    * @param upperBound an upper bound (exclusive) on the values of the randomly generated initial weights
    * @param outputNodes the number of output nodes in the network
    */
   private void setInstanceVariables(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda,
                                     int maxIterations, double stoppingError, String weightsFile, String inputsFile,
                                     String outputsFile, double lowerBound, double upperBound)
   {
      this.layerSizes = new int[hiddenLayerNodes.length+2];     // layerSizes holds input, output, and hidden layers
      this.layerSizes[0] = inputNodes;                          // the first element of layerSizes is the number of input nodes
      this.layerSizes[hiddenLayerNodes.length+1] = outputNodes; // the last element of layerSizes is the number of output nodes
      
      // the interior elements of layerSizes are the lengths of the hidden layer nodes
      for (int i = 1; i <= hiddenLayerNodes.length; i++)
      {
         layerSizes[i] = hiddenLayerNodes[i-1]; // set the value of the layerSizes array to its corresponding value in hiddenLayerNodes
      }
      
      /*
       * Calculate the maximum number of nodes out of the number of input nodes, number of output nodes, and the number of nodes
       * in each of the hidden layers. Uses arrayMax as a helper method which finds the maximum number of nodes in the hidden layer
       * array. Then calls Math.max two more times to find the maximum overall number of nodes.
       */
      int maxNumNodes = Math.max(inputNodes, Math.max(arrayMax(hiddenLayerNodes), outputNodes));

      /*
       * Activations is a rectangular array. The number of rows is the length of layerSizes. 
       * The number of columns is the maximum number of nodes. This will result in a rectangular array
       * instead of a ragged one, which uses more storage but is simpler to use.
       */
      this.activations = new double[layerSizes.length][maxNumNodes];

      /*
       * rawActivations is a rectangular array which is the same dimensions as activations. It stores the activations
       * without an activation function applied to them.
       */
      this.rawActivations = new double[layerSizes.length][maxNumNodes];

      /*
       * Weights is a 3D array. The first dimension is the value of m (the index of the connectivity layer). The number of
       * connectivity layers is 1 more than the number of hidden layers, so that is the value of the first dimension.
       * The second and third dimensions are simply the maximum number of nodes. Like in the activations 2D array
       * we avoid using ragged arrays.
       */
      this.weights = new double[hiddenLayerNodes.length+1][maxNumNodes][maxNumNodes];
      
      /*
       * The omega 2D array has the same dimensions as activations and rawActivations. 
       * Omega values are intermediate values used during backpropagation.
       */
      this.omega = new double[layerSizes.length][maxNumNodes];
      
      /*
       * The psi 2D array also has the same dimensions as activations and rawActivations.
       * Psi values are intermediate values used during backpropagation.
       */
      this.psi = new double[layerSizes.length][maxNumNodes];

      this.lambda = lambda;                               // Set the instance variable lambda
      this.maxIterations = maxIterations;                 // Set the instance variable maxIterations
      this.stoppingError = stoppingError;                 // Set the stopping error threshold
      this.theoreticalOutputs = readOutputs(outputsFile); // Returns a 2D array with the theoretical outputs for all output nodes & cases
      this.lowerBound = lowerBound;                       // Set the instance variable lowerBound
      this.upperBound = upperBound;                       // Set the instance variable upperBound

      readWeights(weightsFile); // call readWeights to read weights (or generate them randomly)
      readInputs(inputsFile);   // call readInputs to read inputs
   }

   /**
    * A helper method that reads theoretical outputs from a specified file name. The method is capable of reading the outputs for each combination
    * of case and node.
    * 
    * @param outputsFile a file name of the text file that specifies the theoretical outputs. The first line in the outputsFile should consist
    *        of two space-separated natural numbers. The first number specifies the number of outputs (ex: 3 if OR, AND, and XOR are the different
    *        output nodes). The second number specifies the number of cases per output (ex: 4 if the pairs 0,0; 0,1; 1,0; and 1,1 are being used
    *        as boolean logic input cases). For example, if the user wants to do OR, and, and XOR as the three outputs on all 4 input pairs, the
    *        outputsFile should look like this:
    *        3 4
    *        0 0 0
    *        1 0 1
    *        1 0 1
    *        1 1 0
    * @precondition the theoretical outputs file accounts for at least one output node and at least one case. If this is not satisfied, a relevant
    *               RuntimeException with a descriptive error message will be thrown
    *               
    * Special considerations: this method performs exception catching to catch an NumberFormatException, FileNotFoundException,
    * or ArrayIndexOutOfBoundsException that may be thrown. It will throw a RuntimeException with a relevant message
    * if either of those occurs
    * 
    * @return a 2D array outputs, which represents the theoretical outputs for each output and case
    */
   private double[][] readOutputs(String outputsFile)
   {
      double[][] outputs; // Declare a double array to store the outputs but do not specify the number of rows and columns (the value is null)

      /*
       * Use a try-catch construct.
       * 
       * It is possible that some contents of the file are not the type they should be (ex: weights cannot be parsed
       * to double). In that case, catch the NumberFormatException and throw a RuntimeException with a relevant message
       * for the user.
       * 
       * It is also possible that the file is misspecified and cannot be read. In that case, catch the
       * FileNotFoundException and throw a RuntimeException with a relevant message for the user.
       * 
       * It is also possible that when the space-separated values are split into an array and the array is read
       * from, the array index will be accessed out of bounds. In that case, catch the ArrayIndexOutOfBoundsException
       * and throw a RuntimeException with a relevant message for the user.
       */
      try
      {
         File myFile = new File(outputsFile); // Create a File object
         Scanner sc = new Scanner(myFile);    // Create a Scanner to scan the File object

         String firstLine = sc.nextLine();                     // Get the first line
         String[] firstLineArray = firstLine.split(" ");       // Split it by spaces
         int numOutputs = Integer.parseInt(firstLineArray[0]); // The first element is the number of output nodes

         // If the number of outputs here does not match the number specified in the configuration file, throw a RuntimeException
         if (numOutputs!=layerSizes[layerSizes.length-1])
         {
            throw new RuntimeException("Number of outputs in configuration file is inconsistent with the specified truth table values");
         }

         int numCases = Integer.parseInt(firstLineArray[1]); // The second element is the number of cases

         if (numOutputs == 0 || numCases == 0) // Nonsensical to run the network if no outputs or cases are specified
         {
            throw new RuntimeException("Number of outputs or cases in the specified theoretical values file was 0");
         }

         outputs = new double[numCases][numOutputs]; // Instantiate outputs with numCases as the first dimension and numOutputs as the second

         for (int i = 0; i < numCases; i++) // Iterate over the number of cases
         {
            String nextLine = sc.nextLine();              // Get the next line
            String[] nextLineArray = nextLine.split(" "); // Split it by spaces

            for(int j = 0; j < numOutputs; j++) // Iterate over the number of outputs per case
            {
               outputs[i][j] = Double.parseDouble(nextLineArray[j]); // Parse a value corresponding to a case & output and put it into the 2d array
            }
         } // for (int i = 0; i < numCases; i++)

         sc.close(); // Close the scanner
      } // try
      catch (NumberFormatException n)
      {
         // Throw RuntimeException if one of the values cannot be parsed as a double
         throw new RuntimeException("Could not parse a weight value as a double");
      }
      catch (FileNotFoundException f)
      {
         // Throw RuntimeException if the file cannot be found
         throw new RuntimeException("The file could not be found");
      }
      catch (ArrayIndexOutOfBoundsException a)
      {
         // Throw RuntimeException if an array index is out of bounds
         throw new RuntimeException("Array index out of bounds. Please check the space-separated values in the configuration file");
      }

      return outputs; // Return the 2D array of outputs
   }

   /**
    * A static helper method that finds the maximum value of an integer array. This is used in the code to find
    * the maximum number of nodes in the hidden layer array.
    * 
    * @param arr an array where the maximum value will be determined
    * @return the maximum value in the array arr
    */
   private static int arrayMax(int[] arr)
   {
      /*
       * Standard way to find max value. Start with max as Integer.MIN_VALUE and for every value in the array,
       * update max to be the max of itself and the value in the array.
       */
      int max = Integer.MIN_VALUE;

      for (int i : arr) // Iterate over each value in the array
      {
         max = Math.max(i, max); // Use the Math.max method to find the max between i and max
      }

      return max; // Return the max value
   }

   /**
    * Reads in the weights from a text file OR generates random weights, depending on whether filename is a path to a file of weights
    * or the word "randomize". If the word "randomize" is used, then weights will be generated randomly using the setRandomWeights
    * helper method If the filename is a path to a weights file, the weights will be read from that text file. In the text file,
    * the weights are whitespace delimited and each new line represents a different value for the connectivity layer index (m).
    * For example, for a 2-2-1 network, the text file will be structured as follows: 
    * w000 w001 w010 w011
    * w100 w110
    * 
    * @param filename the path to a file that will be read OR the word "randomize"
    * @precondition filename is either a file name or the word "randomize"
    * 
    * Special considerations: this method performs exception catching to catch an InputMismatchException
    * or FileNotFoundException that may be thrown. It will throw a RuntimeException with a relevant message
    * if either of those occurs
    */
   private void readWeights(String filename)
   {
      if (filename.equals("randomize")) // Use randomly generated weights
      {
         setRandomWeights();
      }
      else // Use the weights from the file
      {
         /*
          * Use a try-catch construct.
          * It is possible that a weight cannot be parsed as a double. In that case, catch the InputMismatchException
          * and throw a new RuntimeException with an appropriate error message.
          * 
          * It is also possible that the file path is incorrect. In that case, catch the FileNotFoundException
          * and throw a new RuntimeException with an appropriate error message.
          */
         try
         {
            File myFile = new File(filename); // Create a File object
            Scanner sc = new Scanner(myFile); // Create a Scanner to scan the File object

            /*
             * The outermost for loop is going over each connectivity layer. The number of connectivity
             * layers is the number of total layers minus 1.
             */
            for (int m = 0; m < layerSizes.length-1; m++)
            {
               // Use a for loop to iterate an amount of times equal to the number of activations in the previous layer
               for (int prev = 0; prev < layerSizes[m]; prev++)
               {
                  // Use a for loop to iterate an amount of times equal to the number of activations in the next layer
                  for (int next = 0; next < layerSizes[m+1]; next++)
                  {
                     // If there are more items that the Scanner can read from the weights file
                     if (sc.hasNext())
                     {
                        /*
                         * Read the next item as a double, and set it in the weights 3D array
                         * If the item cannot be parsed as a double, an InputMismatchException will be thrown and caught.
                         */
                        weights[m][prev][next] = sc.nextDouble();
                     }
                  } // for (int next = 0; next < layerSizes[m+1]; next++)
               } // for (int prev = 0; prev < layerSizes[m]; prev++)
            } // for (int m = 0; m < layerSizes.length-1; m++)

            sc.close(); // close the scanner object

            System.out.println();

            // The Arrays.deepToString method prints a 2D array
            System.out.println("WEIGHTS (read from file): " + Arrays.deepToString(weights));
         } // try
         catch (InputMismatchException i)
         {
            // Throw RuntimeException if one of the values cannot be parsed as a double
            throw new RuntimeException("Could not parse a weight value as a double");
         }
         catch (FileNotFoundException f)
         {
            // Throw RuntimeException if the file cannot be found
            throw new RuntimeException("The file could not be found");
         }
      } // else
   }

   /**
    * Sets the weights to random values between a lower and upper bound (instance variables). Uses generateRandom()
    * as a helper method to generate random values between a lower and upper bound.
    */
   public void setRandomWeights()
   {
      /*
       * The outermost for loop is going over each connectivity layer. The number of connectivity
       * layers is the number of total layers minus 1.
       */
      for (int m = 0; m < layerSizes.length-1; m++)
      {
         // Use a for loop to iterate an amount of times equal to the number of activations in the previous layer
         for (int prev = 0; prev < layerSizes[m]; prev++)
         {
            // Use a for loop to iterate an amount of times equal to the number of activations in the next layer
            for (int next = 0; next < layerSizes[m+1]; next++)
            {
               // Generate a random weight using generateRandom() and set it in the appropriate place in the weights 3D array
               weights[m][prev][next] = generateRandom();
            }
         }
      } // for (int m = 0; m < layerSizes.length-1; m++)
   }
   
   /**
    * Generates a random value between a lower and upper bound (instance variables) and returns it.
    * 
    * @return a random value between lowerBound and upperBound
    */
   public double generateRandom()
   {
      return Math.random()*(upperBound-lowerBound) + lowerBound;
   }

   /**
    * Runs the network on data. Takes in a double[] of inputs and set the values of the input nodes to be the values in
    * the inputs array. Runs the initial values of the input nodes through the network. This is done by looking at each
    * node in the hidden layers and output layer, and multiplying the previous activations by the weights running from
    * each previous activation to the "current" node (dot product). Returns an array of doubles. 
    * 
    * @param inputs an array of doubles where each item is an activation state of an input node
    * @param raw true if the raw values of the output layer should be returned (no activation function)
    * @return an array of doubles where each item is an output value
    */
   private double[] runNetwork(double[] inputs, boolean raw)
   {
      for (int i = 0; i < layerSizes[0]; i++) // Iterate a number of times equal to the number of nodes in the input layer
      {
         /*
          * Set the value of the activation state in the input layer to be the corresponding value in the inputs array. Do the same
          * thing for the raw activations double array.
          */
         activations[0][i] = inputs[i];
         rawActivations[0][i] = inputs[i];
      } // for (int i = 0; i < layerSizes[0]; i++)

      for (int layer = 1; layer < layerSizes.length; layer++) // Iterate over the different activation layers, excluding the input layer
      {
         for (int node = 0; node < layerSizes[layer]; node++) // Within each layer, iterate over the indices of the nodes
         {
            activations[layer][node] = 0.0;    // Reset the activation value to 0.0
            rawActivations[layer][node] = 0.0; // Do same for the raw Activations double array
            
            double[] prevActivations = activations[layer-1]; // Extract the current activations in the layer to the left of the node
            
            for (int w = 0; w < layerSizes[layer-1]; w++) // The variable w is the left index for the weights
            {
               double prevWeight = weights[layer-1][w][node]; // Get one of the weights to the left

               // Multiply it by the corresponding current activation state to the left, and add that to the current node
               rawActivations[layer][node]+=prevActivations[w]*prevWeight;
               activations[layer][node]+=prevActivations[w]*prevWeight;
            } // for (int w = 0; w < layerSizes[layer-1]; w++)

            // Apply the activation function to the new activation state by calling the activationFunction method
            activations[layer][node] = activationFunction(activations[layer][node]);
         } // for (int node = 0; node < layerSizes[layer]; node++)
      } // for (int layer = 1; layer < layerSizes.length; layer++)

      double[] outputs; // Declare outputs

      if (raw == false)
      {
         /*
          * Extract the array of outputs into a variable so that it can be returned. The Arrays.copyOfRange method allows
          * for extracting a slice of an array. Here, because activations is a rectangular (non-ragged) array, we don't need
          * the whole of the last layer (activations[activations.length-1]). All that is needed is the part from 0 to
          * layerSizes[layerSizes.length-1] (the number of output nodes). Note that the third parameter, which specifies
          * the "to" index, is exclusive.
          */
         outputs = Arrays.copyOfRange(activations[activations.length-1], 0, layerSizes[layerSizes.length-1]);
      } // if (raw == false)
      else
      {
         // Repeat the process above but using the raw activations array
         outputs = Arrays.copyOfRange(rawActivations[rawActivations.length-1], 0, layerSizes[layerSizes.length-1]);
      }

      return outputs; // Return outputs, which was saved earlier
   }

   /**
    * The readInputs method reads the user inputs from a file and sets them to an instance variable.
    * The file must follow a specific format. The first line in the file consists of two whitespace delimited
    * positive integers. The first number is the number of cases and the second is the number of inputs
    * per case. Each line in the file after the first consists of whitespace delimited inputs.
    * Different sets of inputs occur on different lines. For example, a file with n+1 lines would have n sets 
    * of inputs to be run through the network.
    * 
    * @param filename the path of the file to be read
    * 
    * Special considerations: this method performs exception catching to catch an NumberFormatException, FileNotFoundException,
    * or ArrayIndexOutOfBoundsException that may be thrown. It will throw a RuntimeException with a relevant message
    * if either of those occurs
    */
   private void readInputs(String filename)
   {
      int inputsIndex = 0; // An index into the inputs 2D array

      /*
       * Use a try-catch construct.
       * 
       * It is possible that some contents of the file are not the type they should be (ex: inputs cannot be parsed
       * to doubles). In that case, catch the NumberFormatException and throw a RuntimeException with a relevant message
       * for the user.
       * 
       * It is also possible that the file is misspecified and cannot be read. In that case, catch the
       * FileNotFoundException and throw a RuntimeException with a relevant message for the user.
       * 
       * It is also possible that when the space-separated values are split into an array and the array is read
       * from, the array index will be accessed out of bounds. In that case, catch the ArrayIndexOutOfBoundsException
       * and throw a RuntimeException with a relevant message for the user.
       */
      try
      {
         File myFile = new File(filename); // Create a File object
         Scanner sc = new Scanner(myFile); // Create a Scanner to scan the File object

         String firstLine = sc.nextLine();               // Get the first line
         String[] firstLineArray = firstLine.split(" "); // Split it by spaces

         int numCases = Integer.parseInt(firstLineArray[0]);         // Parse the first element as an int, it is the number of cases
         int numInputsPerCase = Integer.parseInt(firstLineArray[1]); // Parse the second element as an int, it is the number of inputs per case
         trainingCases = new double[numCases][numInputsPerCase];     // Instantiate inputs as a 2D array

         System.out.println("INPUTS:");

         while (sc.hasNextLine()) // Keep reading while the scanner can read another line
         {
            String line = sc.nextLine();            // Extract the next line
            String[] splitString = line.split(" "); // Split it by spaces

            /*
             * If the number of elements in the new array (which results from splitting the line by whitespaces),
             * does not match the number of inputs that the network is configured for, then throw a RuntimeException
             * with an appropriate error message.
             */
            if (splitString.length!=layerSizes[0])
            {
               throw new RuntimeException("Incorrect number of inputs specified in the file");
            }

            double[] splitDouble = new double[splitString.length]; // Create a double[] with same length as the String[]

            for (int i = 0; i < splitString.length; i++) // Iterate over the String array
            {
               // Parse the element as a Double and put it into the corresponding slot in the double array
               splitDouble[i] = Double.parseDouble(splitString[i]);
            }

            System.out.println("Input case: " + Arrays.toString(splitString)); // Print out the array of inputs using the Arrays.toString method

            trainingCases[inputsIndex] = splitDouble; // The inputs 2D array at the inputsIndex is set to the next case inputs
            inputsIndex++;                            // Increment inputsIndex because we are moving to the next case
         } // while (sc.hasNextLine())

         System.out.println(); // Add a new line after the inputs are all printed out

         sc.close(); // Close the scanner
      } // try
      catch (NumberFormatException n)
      {
         // Throw RuntimeException if a parsing error occurs
         throw new RuntimeException("Could not parse the file");
      }
      catch (FileNotFoundException f)
      {
         // Throw RuntimeException if the file cannot be found
         throw new RuntimeException("The file could not be found");
      }
      catch (ArrayIndexOutOfBoundsException a)
      {
         // Throw RuntimeException if an array index is out of bounds
         throw new RuntimeException("Array index out of bounds. Please check the space-separated values in the configuration file");
      }
   }

   /**
    * This static method applies an activation function to a given double. It can be changed to different
    * activation functions as the user wishes. The activation function takes a large input and "scales" it
    * down to a input with a much smaller magnitude.
    * 
    * @param x a double value which the activation function will be applied to
    */
   private static double activationFunction(double x)
   {
      return 1.0/(1.0+Math.exp(-x));
   }

   /**
    * This static method applies the derivative of an activation function to a given double. It can be changed
    * as the user wishes.
    * 
    * @param x a double value which the activation function will be applied to
    */
   private static double activationFunctionDerivative(double x)
   {
      double sig = activationFunction(x);
      return sig*(1.0-sig);
   }

   /**
    * Updates weights using the backpropagation algorithm. Uses the 2D array instance variables omega and psi.
    * Works for any number of hidden layers. Uses the mathematical results found in Dr. Nelson's notes: "4-Three Plus Layer Network."
    *
    * @param theoretical an array of theoretical outputs
    * @param calculated an array of calculated outputs
    */
   public void updateWeightsBackprop(double[] theoretical, double[] calculated)
   {
      int lastLayer = layerSizes.length - 1; // Hold the index of the output layer in a variable
      
      for (int lastLayerNode  = 0; lastLayerNode < layerSizes[lastLayer]; lastLayerNode++) // Iterate over output layer
      {
         omega[lastLayer][lastLayerNode] = theoretical[lastLayerNode] - calculated[lastLayerNode]; // Difference between theoretical & calculated
         
         // Psi at each node is the product of omega and the activation function's derivative evaluated at the node
         psi[lastLayer][lastLayerNode] = omega[lastLayer][lastLayerNode]*activationFunctionDerivative(rawActivations[lastLayer][lastLayerNode]);
      } // for (int lastLayerNode  = 0; lastLayerNode < layerSizes[lastLayer]; lastLayerNode++)
      
      for (int layer = layerSizes.length - 2; layer >= 0; layer--) // Iterate over the rest of the layers
      {
         for (int node = 0; node < layerSizes[layer]; node++) // Iterate over the nodes in each layer
         {
            omega[layer][node] = 0.0; // Set omega in the 2D array at the current node to be 0.0
            
            for (int nextLayerNode = 0; nextLayerNode < layerSizes[layer+1]; nextLayerNode++)
            {
               // Get psi at a node to the right times the weight connecting the current node to that node and add to running total omega
               omega[layer][node] += psi[layer+1][nextLayerNode]*weights[layer][node][nextLayerNode];
            }
            
            psi[layer][node] = omega[layer][node] * activationFunctionDerivative(rawActivations[layer][node]); // Same psi calculation as before
            
            for (int nextLayerNode = 0; nextLayerNode < layerSizes[layer+1]; nextLayerNode++) // Iterate over the nodes in the layer to the right
            {
               // Update weight connecting the current node and a node in the layer to the right
               weights[layer][node][nextLayerNode] += lambda*activations[layer][node]*psi[layer+1][nextLayerNode];
            }
         } // for (int node = 0; node < layerSizes[layer]; node++)
      } // for (int layer = layerSizes.length - 2; layer >= 0; layer--)
   }

   /**
    * The gradientDescent method minimizes the total error function by stepping the weights in the opposite direction
    * of the gradient (the direction of steepest ascent). Backpropagation is used to speed up training time and use
    * less resources. See Dr. Nelson's course notes for the mathematical results used here.
    * The network will stop running if the total error drops below the stoppingError threshold, or if the number of iterations
    * reaches maxIterations. The method will also print out the input cases, the results
    * (inputs, theoretical outputs, and actual outputs), and network configuration and information
    * (number of iterations, stopping error threshold, reason for stopping, value of lambda,
    * and final total error.
    */
   public void gradientDescent()
   {
      long start = System.nanoTime(); // Get the current start time in nanoseconds as a long integer
      
      int numIterations = 0; // Counter for the number of iterations

      double[] errorArr = new double[trainingCases.length]; // Initialize an array to store the case errors
      double totalError = Integer.MAX_VALUE;                // Start the totalError with Integer.MAX_VALUE so next ones will be smaller

      while (totalError >= stoppingError && numIterations < maxIterations) // Ends if totalError is below stoppingError or max iterations is reached
      {
         for (int i = 0; i < trainingCases.length; i++) // Iterate over the 2D array of training cases
         {
            double[] myTrainingCase = trainingCases[i];                      // Extract one training case
            //System.out.println("Case: " + Arrays.toString(myTrainingCase));
            double[] theoreticalOutputsArray = theoreticalOutputs[i];        // Get the corresponding theoretical outputs array
            //System.out.println("Theoretical: " + Arrays.toString(theoreticalOutputsArray));
            double[] actualOutputsArray = runNetwork(myTrainingCase, false); // Run the network to get the actual outputs array
            //System.out.println("Actual: " + Arrays.toString(actualOutputsArray));

            updateWeightsBackprop(theoreticalOutputsArray, actualOutputsArray); // Update the model weights according to the design document formulas

            /*
             * Note: Because lambda is fixed and NOT adaptive, we don't need to calculate the case errors at each iteration. If instead, lambda
             * was adaptive, we would calculate the case errors so we could see whether they increased or decreased and adjust lambda accordingly.
             */

            lambda*=LAMBDA_MULTIPLIER; // Adaptive lambda is not implemented, but this is a placeholder in case it is implemented later

            errorArr[i] = calculateError(theoreticalOutputsArray, actualOutputsArray); // Save the case error into an array element
         } // for (int i = 0; i < trainingCases.length; i++)
         totalError = calculateTotalError(errorArr); // Calculate the total error

         numIterations++; // Increment iteration counter
      } // while (totalError >= stoppingError && numIterations < maxIterations)

      System.out.println("RESULTS:"); // Print out the label for the inputs, theoretical and actual outputs

      double[] actualOutputsArray; // Declare actualOutputsArray but don't give it a value

      for (int i = 0; i < trainingCases.length; i++) // Iterate over the trainingCases 2D array
      {
         double[] myTrainingCase = trainingCases[i];               // Extract one training case
         double[] theoreticalOutputsArray = theoreticalOutputs[i]; // Find the theoretical outputs array
         actualOutputsArray = runNetwork(myTrainingCase, false);   // Run the network to get the actual output layer

         errorArr[i] = calculateError(theoreticalOutputsArray, actualOutputsArray); // Save the case error into an array element

         System.out.println("Inputs: " + Arrays.toString(myTrainingCase) + " ");                 // Print out the inputs for a case
         System.out.println("Theoretical outputs: " + Arrays.toString(theoreticalOutputsArray)); // Print out the theoretical output for a case
         System.out.println("Actual outputs: " + Arrays.toString(actualOutputsArray));           // Print out the actual output for a case

      } // for (int i = 0; i < trainingCases.length; i++)
      double finalTotalError = calculateTotalError(errorArr); // Get the final total error of the Perceptron
      
      long end = System.nanoTime();            // Get the current time in nanoseconds as a long integer
      long elapsedMilli = (end-start)/1000000; // Divide the difference in nanosecond times by 10^6 to get milliseconds

      String stoppingReason; // Declare a String to store the reason that the network stopped
      if (numIterations == maxIterations) // The network stopped because numIterations reached maxIterations
      {
         stoppingReason = "max number of iterations reached"; // Set stoppingReason to the relevant reason
      }
      else // Otherwise, it must have stopped before reaching maxIterations (because the error fell below the stopping threshold)
      {
         stoppingReason = "error fell below stopping error theshold"; // Set stoppingReason to the relevant reason
      }

      System.out.println();                                              // Print a new line to separate the results and configuration printing
      System.out.println("NETWORK CONFIGURATION & INFORMATION:");        // Print out the label for the print statements that follow
      System.out.println("Max number of iterations: " + maxIterations);  // Print out the maximum number of iterations
      System.out.println("Stopping error threshold: " + stoppingError);  // Print out the stopping error threshold
      System.out.println("Lambda (fixed): " + lambda);                   // Print out the lambda value
      System.out.println("Total error: " + finalTotalError);             // Print out the final total error
      System.out.println("Number of iterations run: " + numIterations);  // Print out the number of iterations
      System.out.println("Reason for stopping: " + stoppingReason);      // Print out the reason for stopping
      System.out.println("Elapsed training time (ms): " + elapsedMilli); // Print out the elapsed training time in milliseconds
   }

   /**
    * Calculates the error between a theoretical outputs array and actual outputs array according to the formulas
    * in the design document.
    * 
    * @param theoreticalOutputs the theoretical value of outputs
    * @param actualOutputs the actual values of the outputs
    */
   private double calculateError(double[] theoreticalOutputs, double[] actualOutputs)
   {
      double total = 0.0; // Set the total to 0.0

      for (int i = 0; i < theoreticalOutputs.length; i++) // Iterate over each of the output items (could have iterated over actualOutputs instead)
      {
         double difference = theoreticalOutputs[i] - actualOutputs[i]; // Find the difference between the theoretical and actual outputs
         total+=(difference*difference);                               // Add the squared difference to the running total
      }

      return 0.5 * total; // Return half of the total
   }

   /**
    * Calculates the total error in an array of case errors. Each element in the array is squared, and the
    * squares are added together. Then, the square root of the total is returned.
    * 
    * @param errorArr an array where each element is a case error
    */
   private double calculateTotalError(double[] errorArr)
   {
      double total = 0.0; // Set the total to 0.0

      for (double d : errorArr) // Use for each loop to iterate over the case error array
      {
         total += d*d; // Add the square of the case error to the running total
      }

      return Math.sqrt(total); // Return the square root of the total
   }
}