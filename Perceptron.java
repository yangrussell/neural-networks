import java.io.*;
import java.util.*;
/**
 * The Perceptron class implements a feed-forward neural network with a configurable number
 * of input nodes, output nodes, number of hidden layers, and number of nodes in each 
 * hidden layer. A Perceptron object can read inputs and weights from a file and run
 * the network.
 * 
 * Methods:
 * public Perceptron(int inputNodes, int[] hiddenLayerNodes, int outputNodes)
 * private static int arrayMax(int[] arr)
 * private void readWeights(String filename)
 * private double[] runNetwork(double[] inputs)
 * private void resetActivations()
 * private void readInputs(String filename)
 * private static double activationFunction(double x)
 * public static void main(String[] args)
 *
 * @author Russell Yang
 * @version 9/4/2019 (creation date)
 */
public class Perceptron
{
   // instance variables  
   private int[] layerSizes;       // A 1D array representing the sizes of each of the layers
   private double[][] activations; // A 2D array representing the activation states of the network
   private double[][][] weights;   // A 3D array representing the weights of the network

   /**
    * Constructor for the Perceptron class. Sets instance variables to values based on the parameters.
    * 
    * @param inputNodes the number of nodes that the network uses to take in inputs
    * @param hiddenLayerNodes an array where each element is the number of nodes in a hidden layer of the network,
    *        and the length of the array is the number of hidden layers
    * @param outputNodes the number of output nodes in the network
    */
   public Perceptron(int inputNodes, int[] hiddenLayerNodes, int outputNodes)
   {
      this.layerSizes = new int[hiddenLayerNodes.length+2];     // layerSizes holds input, output, and hidden layers
      this.layerSizes[0] = inputNodes;                          // the first element of layerSizes is the number of input nodes
      this.layerSizes[hiddenLayerNodes.length+1] = outputNodes; // the last element of layerSizes is the number of output nodes

      // the interior elements of layerSizes are the lengths of the hidden layer nodes
      for (int i = 1; i <= hiddenLayerNodes.length; i++)
      {
         layerSizes[i] = hiddenLayerNodes[i-1];
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
 * Weights is a 3D array. The first dimension is the value of m (the index of the connectivity layer). The number of
 * connectivity layers is 1 more than the number of hidden layers, so that is the value of the first dimension.
 * The second and third dimensions are simply the maximum number of nodes. Like in the activations 2D array
 * we avoid using ragged arrays.
 */
      this.weights = new double[hiddenLayerNodes.length+1][maxNumNodes][maxNumNodes];
   }

   /**
    * A static helper method that finds the maximum value of an array. This is used in the code to find
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

      for (int i : arr)
      {
         max = Math.max(i, max);
      }

      return max;
   }

   /**
    * Reads in the weights from a text file. In the text file, the weights are whitespace delimited and each
    * new line represents a different value for the connectivity layer index (m). For example, for a 2-2-1 network,
    * the text file will be structured as follows: 
    * w000 w001 w010 w011
    * w100 w110
    * 
    * Special considerations: this method performs exception catching to catch an InputMismatchException
    * or FileNotFoundException that may be thrown. It will throw a RuntimeException with a relevant message
    * if either of those occurs
    * 
    * @param filename the path to a file that will be read
    */
   private void readWeights(String filename)
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
                  } // if (sc.hasNext())
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
   }

   /**
    * Runs the network on data. Takes in a double[] of inputs and set the values of the input nodes to be the values in
    * the inputs array. Runs the initial values of the input nodes through the network. This is done by looking at each
    * node in the hidden layers and output layer, and multiplying the previous activations by the weights running from
    * each previous activation to the "current" node. Calls the helper method resetActivations to set all the activations
    * to 0 after the input data is run through the network. Returns an array of doubles 
    * 
    * @param inputs an array of doubles where each item is an activation state of an input node
    * @return an array of doubles where each item is an output value
    */
   private double[] runNetwork(double[] inputs)
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

            // Apply the activation function to the new activation state by calling the activationFunction method
            activations[layer][node] = activationFunction(activations[layer][node]);
         } // for (int node = 0; node < layerSizes[layer]; node++)
      } // for (int layer = 1; layer < layerSizes.length; layer++)

/*
 * Extract the array of outputs into a variable so that it can be returned. The Arrays.copyOfRange method allows
 * for extracting a slice of an array. Here, because activations is a rectangular (non-ragged) array, we don't need
 * the whole of the last layer (activations[activations.length-1]). All that is needed is the part from 0 to
 * layerSizes[layerSizes.length-1] (the number of output nodes). Note that the third parameter, which specifies
 * the "to" index, is exclusive.
 */
      double[] outputs = Arrays.copyOfRange(activations[activations.length-1], 0, layerSizes[layerSizes.length-1]);

      // Call a method to set all the activation states back to 0
      resetActivations();

      // Return outputs, which was saved earlier
      return outputs;
   }

   /**
    * The resetActivations method resets all of the activation states to 0.0, so that the network
    * can be run on new data.
    */
   private void resetActivations()
   {
      for (int i = 0; i < activations.length; i++) // Iterate over the rows
      {
         for (int j = 0; j < activations[0].length; j++) // Iterate over the columns
         {
            activations[i][j] = 0.0; // Set the activation state to 0.0
         }
      }
   }

   /**
    * The readInputs method reads the user inputs from a file. The file must follow a specific format.
    * Each line in the file consists of whitespace delimited inputs. Different sets of inputs occur
    * on different lines. For example, a file with n lines would have n sets of inputs to be run
    * through the network.
    * 
    * Special considerations: this method performs exception catching to catch a NumberFormatException
    * or FileNotFoundException that may be thrown. It will throw a RuntimeException with a relevant message
    * if either of those occurs
    * 
    * @param filename the path of the file to be read
    */
   private void readInputs(String filename)
   {
/*
 * Use a try-catch construct.
 * It is possible that after splitting each line and attempting to convert each element into a double,
 * an element cannot be converted. A NumberFormatException will be thrown. Catch that exception
 * and throw a new RuntimeException with an appropriate error message.
 * 
 * It is also possible that the file path is incorrect. In that case, catch the FileNotFoundException
 * and throw a new RuntimeException with an appropriate error message.
 */
      try
      {
         File myFile = new File(filename); // Create a File object
         Scanner sc = new Scanner(myFile); // Create a Scanner to scan the File object

         // Keep reading while the scanner can read another line
         while (sc.hasNextLine())
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

            // Create a double[] with same length as the String[]
            double[] splitDouble = new double[splitString.length];

            // Iterate over the String array
            for (int i = 0; i < splitString.length; i++)
            {
               // Parse the element as a Double and put it into the corresponding slot in the double array
               splitDouble[i] = Double.parseDouble(splitString[i]);
            }

            // Print out the array of inputs using the Arrays.toString method
            System.out.println("INPUT(S): " + Arrays.toString(splitString));

/*
 * Run the network on the current set of inputs and print out the outputs. Before the outputs
 * are actually printed out, the activation states will be, because runNetwork is called on the splitDouble array,
 * which prints out the activation states as the inputs travel through the network. Because runNetwork returns
 * an array of outputs, that array will then be printed out.
 */
            System.out.println("OUTPUT(S): " + Arrays.toString(runNetwork(splitDouble)));

            // Print out a new line so when the code inside the while loop runs again, the output will be on a different line
            System.out.println();
         } // while (sc.hasNextLine())

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
   }

   /**
    * This method applies an activation function to a given double. It can be changed to different
    * activation functions, such as the sigmoid function f(x) = 1/(1+Math.exp(-x)) or the identity function f(x) = x.
    * 
    * @param x a double value which the activation function will be applied to
    */
   private static double activationFunction(double x)
   {
      return x; // For now, just return the input
   }

   /**
    * The main method is used to instantiate a Perceptron object, read weights, read inputs (and run the network).
    * 
    * @param args the supplied command-line arguments
    */
   public static void main(String[] args)
   {
/*
 * The network shall have 1 hidden layer with 2 nodes in it. This can be changed to any number of layers
 * and any number of nodes for each layer in the hidden layer nodes array.
 */
      int[] myHiddenLayerNodes = {2};

/*
 * Create a Perceptron object with 2 input nodes, 1 hidden layer with 2 nodes, and 1 output node.
 * Like the number of hidden layers and hidden layer nodes, the number of input and output nodes
 * can be easily changed.
 */
      Perceptron myPerp = new Perceptron(2, myHiddenLayerNodes, 1);

/*
 * Read the weights from the specified file. In this example, the input passed to readWeights is a
 * relative file path because files is a folder containing the project.
 */
      myPerp.readWeights("files/weights.txt");

/*
 * Read the inputs from the specified file and run the network on each set of inputs.
 * In this example, the input passed to readWeights is a relative file path because files is a 
 * folder containing the project.
 */
      myPerp.readInputs("files/inputs.txt");
   }
}