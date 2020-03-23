/*
 *    WriteStreamToARFFFile.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
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
import java.lang.reflect.Array;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import moa.core.DoubleVector;
import moa.core.FastVector;
import moa.streams.InstanceStream;
//import de.jungblut.nlp.VectorizerUtils;

/**
 * Task to output a stream to an ARFF file
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class WriteHashTrick3 extends AuxiliarMainTask {

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
            "n the target dimension of the vector.", 4);
        
        
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
                  
                    List<Double> hashString = hashVectorize(vec,this.nValue.getValue(),
                            Hashing.murmur3_128((int) System.currentTimeMillis()));
                    List<String> hash2 = new ArrayList<String>();
                    for (Double d : hashString){
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
    
    
      public  List<Double> hashVectorize(List<String> inputFeature, int n,
      com.google.common.hash.HashFunction hashFunction) {
        //   System.out.println(inputFeature);
          List<Double> dense = new ArrayList<>(n);
     
          for (int i = 0 ; i < n ; i++) {
              dense.add(0d);
             
          }

          for (String string : inputFeature) {          
      // get the hash as int
      int hash = 0;
              try{
              hash = hashFunction.hashString(string,Charset.defaultCharset()).asInt();
        
              }catch(Exception e) {
              }
      // abs it, as we don't want to have negative index access
    
      int bucket = Math.abs(hash) % n;
        
             //System.out.println(hash);
      // subtract 1 in case of negative values, else increment.
      // this replaces the second hash function proposed by Weinberger et al.
      dense.set(bucket, dense.get(bucket) + (hash < 0 ? -1d : 1d));
        }
    return dense;
    }
      public InstancesHeader getHeader() {
        return this.streamHeader;
      }
      
      protected void generateHeader(  InstanceStream stream ) {
        FastVector attributes = new FastVector();
     
      
        for (int i = 0; i < this.nValue.getValue(); i++) {
            attributes.addElement(new Attribute("numeric" + (i + 1)));
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
