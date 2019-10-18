import java.io.*;
import java.util.*;
/**
 * The Perceptron class implements a feed-forward neural network with a configurable number
 * of input nodes, output nodes, number of hidden layers, and number of nodes in each 
 * hidden layer. A Perceptron object can read inputs and weights from a file and run
 * the network.
 * 
 * Methods:
 * public Perceptron(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda, int maxIterations, String booleanLogic, double lowerBound, double UpperBound)
 * public Perceptron(String filename)
 * private void setInstanceVariables(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda, int maxIterations, String outputsFile, double lowerBound, double upperBound)
 * private double[][] readOutputs(String outputsFile)
 * private static int arrayMax(int[] arr)
 * private void readWeights(String filename)
 * private double[] runNetwork(double[] inputs)
 * private void resetActivations()
 * private double[][] readInputs(String filename)
 * private static double activationFunction(double x)
 * private static double activationFunctionDerivative(double x)
 * private static double inverseActivation(double x)
 * private void updateWeights(double[] theoretical, double[] calculated)
 * private void gradientDescent(double[][] trainingCases)
 * private double[][][] deepCopyWeights(double[][][] weights)
 * private double calculateError(double[] theoreticalOutputs, double[] actualOutputs)
 * private double calculateTotalError(double[] errorArr)
 * public static void main(String[] args)
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
   private double lambda;                               // A value of lambda, the learning factor
   private int maxIterations;                           // A maximum number of iterations that the network will train for
   private double[][] theoreticalOutputs;               // A 2D array - each row (besides first) is a array of outputs for each input case
   private double lowerBound;                           // Lower bound on the random weights
   private double upperBound;                           // Upper bound on the random weights
   private double[][] trainingCases;                    // A 2D array - each row (besides first) is an input case
   private static final double LAMBDA_MULTIPLIER = 1.0; // A value to multiply lambda by (not in use now as the network is not adaptive)

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
   public Perceptron(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda, int maxIterations, String weightsFile, String trainingCases, String outputsFile, double lowerBound, double upperBound)
   {
      // Call the setInstanceVariables method to set the instance variables to the passed values
      setInstanceVariables(inputNodes, hiddenLayerNodes, outputNodes, lambda, maxIterations, weightsFile, trainingCases, outputsFile, lowerBound, upperBound);
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

         String thirdLine = sc.nextLine();                  // Get third scanner line
         int numOutputNodes = Integer.parseInt(thirdLine);  // Get number of output nodes by parsing the third line to an Integer

         String fourthLine = sc.nextLine();                 // Get fourth scanner line
         double lambda = Double.parseDouble(fourthLine);    // Get lambda by parsing the fourth line to a Double

         String fifthLine = sc.nextLine();                  // Get fifth scanner line
         int maxIterations = Integer.parseInt(fifthLine);   // Get the max number of iterations by parsing the fifth line to an Integer

         String weightsFile = sc.nextLine();                // Get the filename (or the word "randomize") of the weights (the sixth line)

         String inputsFile = sc.nextLine();                 // Get the filename of the inputs file (the seventh line)
         
         String outputsFile = sc.nextLine();                // Get the filename of the outputsFile (the seventh line)

         String eighthLine = sc.nextLine();                 // Get eighth scanner line
         String[] bounds = eighthLine.split(" ");           // Split the eighth scanner line by spaces and save it to an array of Strings
         double lowerBound = Double.parseDouble(bounds[0]); // Get lower bound by parsing the first element of bounds to a Double
         double upperBound = Double.parseDouble(bounds[1]); // Get upper bound by parsing the second element of bounds to a Double

         sc.close(); // close the scanner object

         // Call the setInstanceVariables method to set the instance variables to the values read from the configuration file
         setInstanceVariables(numInputNodes, hiddenLayerNodesArray, numOutputNodes, lambda, maxIterations, weightsFile, inputsFile, outputsFile, lowerBound, upperBound);
      } //try
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
         // Throw RuntimeExceptioon if an array index is out of bounds
         throw new RuntimeException("Array index out of bounds. Please check the space-separated values in the configuration file");
      }
   }

   /**
    * A helper method that sets instance variable values based on values passed from either of the constructors.
    * @param inputNodes the number of nodes that the network uses to take in inputs
    * @param hiddenLayerNodes an array where each element is the number of nodes in a hidden layer of the network,
    *        and the length of the array is the number of hidden layers
    * @param outputNodes the number of nodes in the output layer
    * @param lambda a value of lambda, the learning factor
    * @param maxIterations the maximum number of iterations the network will be trained for
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
   private void setInstanceVariables(int inputNodes, int[] hiddenLayerNodes, int outputNodes, double lambda, int maxIterations, String weightsFile, String inputsFile, String outputsFile, double lowerBound, double upperBound)
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

      this.lambda = lambda;                               // Set the instance variable lambda
      this.maxIterations = maxIterations;                 // Set the instance variable maxIterations
      this.theoreticalOutputs = readOutputs(outputsFile); // readOutputs returns a 2D array with the theoretical outputs for all output nodes & cases
      this.lowerBound = lowerBound;                       // Set the instance variable lowerBound
      this.upperBound = upperBound;                       // Set the instance variable upperBound

      readWeights(weightsFile);
      readInputs(inputsFile);
   }

   /**
    * A helper method that reads theoretical outputs from a specified file name. The method is capable of reading the outputs for each combination
    * of case and node.
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
      } //try
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
         // Throw RuntimeExceptioon if an array index is out of bounds
         throw new RuntimeException("Array index out of bounds. Please check the space-separated values in the configuration file");
      }

      return outputs; // Return outputs, which will be null if 
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
    * or the word "randomize". If the word "randomize" is used, then weights will be generated randomly within a lower and upper bound.
    * If the filename is a path to a weights file, the weights will be read from that text file. In the text file, the weights are whitespace
    * delimited and each new line represents a different value for the connectivity layer index (m). For example, for a 2-2-1 network,
    * the text file will be structured as follows: 
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
                  // Generate a random weight from lowerBound to upperBound at set it in the appropriate place in the weights 3D array
                  weights[m][prev][next] = Math.random()*(upperBound-lowerBound) + lowerBound;
               }
            }
         } // for (int m = 0; m < layerSizes.length-1; m++)
      } // if (filename.equals("randomize"))
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

            /*
             * Print out the weights that were read from the file. For some reason, BlueJ starts the first print statement
             * with a space, so print out a new line first to avoid that.
             */
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
    * Runs the network on data. Takes in a double[] of inputs and set the values of the input nodes to be the values in
    * the inputs array. Runs the initial values of the input nodes through the network. This is done by looking at each
    * node in the hidden layers and output layer, and multiplying the previous activations by the weights running from
    * each previous activation to the "current" node (dot product). Calls the helper method resetActivations to set all the activations
    * to 0 after the input data is run through the network. Returns an array of doubles 
    * 
    * @param inputs an array of doubles where each item is an activation state of an input node
    * @return an array of doubles where each item is an output value
    */
   private double[] runNetwork(double[] inputs, boolean raw)
   {
      // Iterate a number of times equal to the number of nodes in the input layer
      for (int i = 0; i < layerSizes[0]; i++)
      {
         /*
          * Set the value of the activation state in the input layer to be the corresponding value in the inputs array
          */
         activations[0][i] = inputs[i];
      } // for (int i = 0; i < layerSizes[0]; i++)

      // Iterate over the different activation layers, excluding the input layer
      for (int layer = 1; layer < layerSizes.length; layer++)
      {
         // Within each layer, iterate over the indices of the nodes
         for (int node = 0; node < layerSizes[layer]; node++)
         {
            activations[layer][node] = 0.0; // Reset the activation value to 0.0

            // Extract the current activations in the layer to the left of the node
            double[] prevActivations = activations[layer-1];

            // The variable w is the left index for the weights
            for (int w = 0; w < layerSizes[layer-1]; w++)
            {
               // Get one of the weights to the left
               double prevWeight = weights[layer-1][w][node];

               // Multiply it by the corresponding current activation state to the left, and add that to the current node
               activations[layer][node]+=prevActivations[w]*prevWeight;
            } // for (int w = 0; w < layerSizes[layer-1]; w++)

            // Before applying the activation function, copy the raw activations to the rawActivations 2D array
            rawActivations[layer][node] = activations[layer][node];
            
            // Apply the activation function to the new activation state by calling the activationFunction method
            activations[layer][node] = activationFunction(activations[layer][node]);
         } // for (int node = 0; node < layerSizes[layer]; node++)
      } // for (int layer = 1; layer < layerSizes.length; layer++)

      double[] outputs; // Declare outputs
      
      if (raw = false)
      {
         /*
          * Extract the array of outputs into a variable so that it can be returned. The Arrays.copyOfRange method allows
          * for extracting a slice of an array. Here, because activations is a rectangular (non-ragged) array, we don't need
          * the whole of the last layer (activations[activations.length-1]). All that is needed is the part from 0 to
          * layerSizes[layerSizes.length-1] (the number of output nodes). Note that the third parameter, which specifies
          * the "to" index, is exclusive.
          */
         outputs = Arrays.copyOfRange(activations[activations.length-1], 0, layerSizes[layerSizes.length-1]);
      }
      else
      {
         // Repeat the process above but using the raw activations array
         outputs = Arrays.copyOfRange(rawActivations[rawActivations.length-1], 0, layerSizes[layerSizes.length-1]);
      }

      return outputs; // Return outputs, which was saved earlier
   }

   /**
    * The readInputs method reads the user inputs from a file and sets them to an instance variable.
    * The file must follow a specific format. Each line in the file consists of whitespace delimited inputs.
    * Different sets of inputs occur on different lines. For example, a file with n lines would have n sets 
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
         int numInputsPerCase = Integer.parseInt(firstLineArray[1]); // Parse the second element as an int, it is the numbe of inputs per case
         trainingCases = new double[numCases][numInputsPerCase];            // Instantiate inputs as a 2D array

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
            inputsIndex++;                     // Increment inputsIndex because we are moving to the next case
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
         // Throw RuntimeExceptioon if an array index is out of bounds
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
    * Updates the weights according to the theoretical and calculated outputs.
    * 
    * @param theoretical an array of theoretical outputs
    * @param calculated an array of calculated outputs
    */
   private void updateWeights(double[] theoretical, double[] calculated)
   {
      // For the weights with index 1
      for(int outputNode = 0; outputNode < layerSizes[layerSizes.length-1]; outputNode++)
      {
         // Get calculated output
         double output = calculated[outputNode];

         // Take the activation function derivative of the output
         double derivCalculated = activationFunctionDerivative(output);

         // Find the difference between the theoretical and calculated outputs
         double difference = theoretical[outputNode] - calculated[outputNode];
         for (int node = 0; node < layerSizes[1]; node++)
         {
            // Calculate the deltaWeight using the formula
            double deltaWeight = lambda*(difference)*derivCalculated*activations[1][node];

            // Update weight
            weights[1][node][outputNode] += deltaWeight;
         }
      } // for(int outputNode = 0; outputNode < layerSizes[layerSizes.length-1]; outputNode++)

      for (int outputNode = 0; outputNode < layerSizes[layerSizes.length-1]; outputNode++) // For the weights with index 0
      {
         // Get calculated output
         double output = calculated[outputNode];

         // Take the activation function derivative of the output
         double derivCalculated = activationFunctionDerivative(output);

         // Find the difference between the theoretical and calculated outputs
         double difference = theoretical[outputNode] - calculated[outputNode];
         for (int prev = 0; prev < layerSizes[0]; prev++) // Iterate over the previous nodes
         {
            for (int next = 0; next < layerSizes[1]; next++) // Iterate over the next node
            {
               // Get the activation that lies on the right of the weight
               double rightActivation = rawActivations[1][next];

               // Get the activation function derivative of it
               double derivRight = activationFunctionDerivative(rightActivation);

               // Get the delta weight using the formula
               double deltaWeight = lambda*activations[0][prev]*derivRight*(difference)*derivCalculated*weights[1][next][outputNode];

               // Update the weight
               weights[0][prev][next] += deltaWeight;
            } // for (int next = 0; next < layerSizes[1]; next++)
         } // for (int prev = 0; prev < layerSizes[0]; prev++)
      } // for (int outputNode = 0; outputNode < layerSizes[layerSizes.length-1]; outputNode++)
   }

   /**
    * The gradientDescent method minimizes the total error function by stepping the weights in the opposite direction
    * of the gradient (the direction of steepest ascent). See the design document for the mathematical results used here.
    */
   private void gradientDescent()
   {
      int numIterations = 0; // Counter for the number of iterations

      double[] errorArr = new double[trainingCases.length]; // Initialize an array to store the case errors

      while (numIterations < maxIterations) // This will run until numIterations equals maxIterations
      {
         for (int i = 0; i < trainingCases.length; i++) // Iterate over the 2D array of training cases
         {
            double[] myTrainingCase = trainingCases[i];                           // Extract one training case
            double[] theoreticalOutputsArray = theoreticalOutputs[i];             // Get the corresponding theoretical outputs array
            double[] actualOutputsArray = runNetwork(myTrainingCase, true); // Run the network to get the actual outputs array

            updateWeights(theoreticalOutputsArray, actualOutputsArray); // Update the model weights according to the design document formulas

            /*
             * Note: Because lambda is fixed and NOT adaptive, we don't need to calculate the case errors at each iteration. If instead, lambda
             * was adaptive, we would calculate the case errors so we could see whether they increased or decreased and adjust lambda accordingly.
             */
            
            lambda*=LAMBDA_MULTIPLIER; // Adaptive lambda is not implemented, but this is a placeholder in case it is implemented layer
         } // for (int i = 0; i < trainingCases.length; i++)
         
         // Increment iteration counter
         numIterations++;
      } // while (numIterations < maxIterations)

      double totalError = calculateTotalError(errorArr); // Calculate the total error
      System.out.println("RESULTS:");                    // Print out the label for the inputs, theoretical and actual outputs

      for (int i = 0; i < trainingCases.length; i++) // Iterate over the trainingCases 2D array
      {
         double[] myTrainingCase = trainingCases[i];                      // Extract one training case
         double[] theoreticalOutputsArray = theoreticalOutputs[i];        // Find the theoretical outputs array
         double[] actualOutputsArray = runNetwork(myTrainingCase, false); // Run the network to get the actual output layer

         errorArr[i] = calculateError(theoreticalOutputsArray, actualOutputsArray); // Save the case error into an array element
         
         System.out.println("Inputs: " + Arrays.toString(myTrainingCase) + " ");                 // Print out the inputs for a case
         System.out.println("Theoretical outputs: " + Arrays.toString(theoreticalOutputsArray)); // Print out the theoretical output for a case
         System.out.println("Actual outputs: " + Arrays.toString(actualOutputsArray));           // Print out the actual output for a case

      } // for (int i = 0; i < trainingCases.length; i++)
      
      System.out.println();                                                // Print a new line to separate the results and configuration printing
      System.out.println("NETWORK CONFIGURATION & FINAL TOTAL ERROR");     // Print out the label for the print statements that follow
      System.out.println("Number of iterations: " + numIterations);        // Print out the number of iterations
      System.out.println("Lambda (fixed): " + lambda);                     // Print out the lambda value
      System.out.println("Total error: " + calculateTotalError(errorArr)); // Print out the final total error
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

   /**
    * The main method is used to read a configuration file from the user and instantiate a Perceptron object.
    * The configuration file must specify data in the following format:
    * number of input nodes
    * number of hidden layer nodes (put space-separated values if there are multiple hidden layers)
    * number of output nodes
    * value of lambda (fixed)
    * max number of iterations
    * filename for weights file or "randomize" if the weights should be random
    * filename for inputs file
    * filename for outputs file
    * lower bound on random weights upper bound on random weights (space-separated)
    * 
    * For example, for a 2-4-3 multiple outputs network with a lambda of 0.5,
    * a max number of iterations 1000000, random staring weights, input file
    * "files/inputs.txt", output file "files/all.txt", lower bound on lambda
    * -1.0, upper bound on lambda 1.0, the file would look like this:
    * 2
    * 4
    * 3
    * 0.5
    * 1000000
    * randomize
    * files/inputs.txt
    * files/all.txt
    * -1.0 1.0
    * 
    * @param args the supplied command-line arguments
    */
   public static void main(String[] args)
   {
      Scanner sc = new Scanner(System.in);                                           // Make a scanner to accept user input
      System.out.println("Enter a configuration file name (ex: files/config.txt):"); // Ask the user for a configuration file
      String configFile = sc.nextLine();                                             // Read the user's input
      
      Perceptron myPerp = new Perceptron(configFile); // Create a Perceptron object with the user's configuration file

      myPerp.gradientDescent(); // Perform gradient descent to find a minimum error
   }
}