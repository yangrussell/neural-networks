import java.io.*;
import java.util.*;
/**
 * The Main class is used to instantiate and run Perceptron objects. It also interacts with the DibDump file.
 *
 * @author Russell Yang
 * @version 10/31/19 (creation date)
 */
public class Main
{  
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
      System.out.println();
      
      Scanner sc = new Scanner(System.in);                                           // Make a scanner to accept user input
      System.out.println("Enter a configuration file name (ex: files/config.txt):"); // Ask the user for a configuration file
      String configFile = sc.nextLine();                                             // Read the user's input
      
      System.out.println("STEP 1: IMAGE TO FILE");
      DibDump myDibDump = new DibDump();
      DibDumpUtility.imageToFile(myDibDump);
      System.out.println("   Saved the image to file");
      System.out.println();
      
      System.out.println("STEP 2: TRAINING NETWORK");
      Perceptron myPerp = new Perceptron(configFile); // Create a Perceptron object with the user's configuration file

      myPerp.gradientDescent(); // Perform gradient descent to find a minimum error
      
      
      System.out.println("   Trained the network using the file as input activations");
      System.out.println();
   
      System.out.println("STEP 3: OUTPUTS FILE TO IMAGE");
      File f = new File("outputs.txt");
      DibDumpUtility.fileToImage(10, 10, f);
      System.out.println("   Converted the network output file back to an iamge");
   }
}
