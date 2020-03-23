
package moa.tasks;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.Writer;

import moa.core.ObjectRepository;
import moa.options.ClassOption;
import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.google.common.hash.Hashing;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import moa.core.FastVector;
import moa.streams.InstanceStream;
//import de.jungblut.nlp.VectorizerUtils;

/**
 * WriteStreamToARFFFile.java
 * Task to output a stream to an ARFF file
 *
 */
public class CSGaussian extends AuxiliarMainTask {

     @Override
    public String getPurposeString() {
        return "Outputs a stream to an ARFF file.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to write.", InstanceStream.class,
            "generators.RandomTreeGenerator");

    public FileOption arffFileOption = new FileOption("arffFile", 'f',
            "Destination ARFF file.", null, "arff", true);

    public IntOption maxInstancesOption = new IntOption("maxInstances", 'm',
            "Maximum number of instances to write to file.", 10000000, 0,
            Integer.MAX_VALUE);

    public FlagOption suppressHeaderOption = new FlagOption("suppressHeader",
            'h', "Suppress header from output.");

    
    public IntOption nValue = new IntOption("nTargetfeat", 'n',
            "n the target dimension of the vector.", 10);
        
        
    protected InstancesHeader streamHeader;

       
    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
    
        InstanceStream stream = (InstanceStream) getPreparedClassOption(this.streamOption);
        File destFile = this.arffFileOption.getFile(); 
        
        if (destFile != null) {
            try {
                Writer w = new BufferedWriter(new FileWriter(destFile));
                monitor.setCurrentActivityDescription("Writing stream to ARFF");
                if (!this.suppressHeaderOption.isSet()) {
                    generateHeader(stream);
                    w.write(getHeader().toString());
                    w.write("\n");
                }
                int numWritten = 0;
                while ((numWritten < this.maxInstancesOption.getValue())
                        && stream.hasMoreInstances()) {
                    
            
                    List<String> vec = new LinkedList(Arrays.asList(stream.nextInstance().getData().toString().split(",")));
                    String lastElem = vec.get(vec.size()-1);
                    
                    if (vec!=null && !vec.isEmpty() ){
                        vec.remove(vec.size()-1);
                    }
                    
                   // System.out.println(vec);
                  
                    List<Integer> hashString = hashVector(vec,this.nValue.getValue(), Hashing.murmur3_128());
                    List<String> hash2 = new ArrayList<String>();
                    for (Integer d : hashString){
                        hash2.add(d.toString());
                    }
                   
                    //System.out.println(hashString);
                    hash2.add(hashString.size(),lastElem.concat(","));
                    
                    
                    String inse = hash2.toString().replaceAll("\\s","");
                    inse = inse.substring(1, inse.length() - 1);
                    w.write(inse);
                    w.write("\n");
                    numWritten++;
                   
                }
                w.close();
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Failed writing to file " + destFile, ex);
            }
            return "Stream written to ARFF file " + destFile;
        }
        throw new IllegalArgumentException("No destination file to write to.");
    }

    @Override
    public Class<?> getTaskResultType() {
        return String.class;
    }
    
    
      public  List<Integer> hashVector( List<String> inputFeat, int n,
      com.google.common.hash.HashFunction hashFunction) {
          //output vector: dense represnetation of the original values
          List<Integer> dense = new ArrayList<>(n);
          for (int i = 0 ; i < n ; i++) {
                     dense.add(0);
          }
          
           for (int i = 0; i < inputFeat.size() ; i++){
              
              try{ //with boolean and numerical attributes
                  //System.out.println("2");
              double diff = Math.abs(Double.parseDouble(inputFeat.get(i)));
                  //System.out.println("3");
              if( diff  > Double.MIN_NORMAL) {
            
                  //lst.add(inputFeature.get(i)+""+i);
                  int  hash = hashFunction.hashInt(i).asInt();
                  
                // abs it, as we don't want to have negative index access
                    int bucket = Math.abs(hash) % n;
                     //System.out.println(hash);
                     // subtract 1 in case of negative values, else increment.
                    // this replaces the second hash function proposed by Weinberger et al.
                    dense.set(bucket, 1);
              }
              }
              catch(NumberFormatException e){ //categorical attributes
                //  System.out.println("pourqyou");
                  if (!inputFeat.get(i).equals("class1")){ //for tweets which are boolean {class1, class2), if it is different from class1 equals to zero
                      int  hash = hashFunction.hashInt(i).asInt();
                      int bucket = Math.abs(hash) % n;
                      dense.set(bucket, 1);
                  
                  }
                  
              }
          }
          /*     else {
                  int  hash = hashFunction.hashInt(i).asInt();
                  int bucket = Math.abs(hash) % n;
                   dense.set(bucket, dense.get(bucket) + (hash < 0 ? -1d : 1d));
              }
                  }
           */
         
         
    return dense;
    }
      public InstancesHeader getHeader() {
        return this.streamHeader;
      }
      
      protected void generateHeader(  InstanceStream stream ) {
        FastVector attributes = new FastVector();
        String [] attribut = new String []{"0","1"} ; 
      
        for (int i = 0; i < this.nValue.getValue(); i++) {
            attributes.addElement(new Attribute("nominal" + (i + 1),
                    Arrays.asList(attribut)));
        }
        FastVector classLabels = new FastVector();
       
        List<String> cl = new ArrayList( stream.getHeader().classAttribute().getAttributeValues());
       
         for (String c : cl){
                        classLabels.add(c);
                    }
     // String cla = cl.toString().replaceAll("\\s","");
       // cla = cla.substring(1, cla.length() - 1);
        
      //classLabels.add(cla);
        
        
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
    }
    }
