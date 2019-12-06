import java.io.*;
import java.util.*;
/**
 * DibDumpUtility is a class which utilizes the DibDump class to manipulate bitmaps. It can convert a bitmap to a file
 * or a file to a bitmap.
 *
 * @author Russell Yang
 * @version (a version number or a date)
 */
public class DibDumpUtility
{
   private static final double MAX_PRENORMALIZED = 2147483648.0;
   
   public static void imageToFile(String filename)
   {
      DibDump myDibDump = new DibDump();
      myDibDump.main(new String[]{});
      
      File out = new File(filename);
      PrintStream stream = null;

      try
      {
         out.createNewFile();
         stream = new PrintStream(out);
      }
      catch (IOException i)
      {
         throw new RuntimeException("An error occured while writing the file");
      }

      int[][] imageArray = myDibDump.imageArray;

      for (int i = 0; i < imageArray.length; i++)
      {
         for (int j = 0; j < imageArray[i].length; j++)
         {
            stream.print(((double)(myDibDump.swapInt(imageArray[i][j])))/MAX_PRENORMALIZED + " ");
         }
         //stream.println();
      }
   }

   public static void fileToImage(int rowLength, int colLength, String filename)
   {
      int[][] imageArray = new int[rowLength][colLength];
      File myPels = new File(filename);
      
      Scanner sc;
      try
      {
         sc = new Scanner(myPels);
      }
      catch (FileNotFoundException f)
      {
         throw new RuntimeException("The pels file could not be found");
      }

      DibDump myDibDump = new DibDump(imageArray);
      
      for (int i = 0; i < imageArray.length; i++)
      {
         for (int j = 0; j < imageArray[i].length; j++)
         {
            myDibDump.imageArray[i][j] = myDibDump.swapInt((int)(MAX_PRENORMALIZED*sc.nextDouble()));
         }
      }
      myDibDump.writeFile();
   }
}
