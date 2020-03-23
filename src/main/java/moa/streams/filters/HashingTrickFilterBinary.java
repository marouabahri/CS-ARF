

package moa.streams.filters;
import com.github.javacliparser.IntOption;
import com.google.common.hash.Hashing;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import java.util.Arrays;
import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.streams.InstanceStream;

/**
 *    Filter to perform feature hashing, to reduce the number of attributes. It applies
 *    a hash function to the features and using their hash values as indices directly,
 *    rather than looking the indices up in an associative array.
 *
 * @author Maroua Bahri
 */

public class HashingTrickFilterBinary extends AbstractStreamFilter {

    private static final long serialVersionUID = 1L;

    public IntOption dim = new IntOption("FeatureDimension", 'd',
            "the target feature dimension.", 10);

    protected InstancesHeader streamHeader;

    protected FastVector attributes;

    @Override
    protected void restartImpl() {
        this.streamHeader = null;
    }

    @Override
    public InstancesHeader getHeader() {
        return this.streamHeader;
    }

    @Override
    public InstanceExample nextInstance() {
        Instance sparseInstance = (Instance) this.inputStream.nextInstance().getData();

        if (streamHeader == null) {
            //Create a new header
            this.attributes = new FastVector();
       
           String [] attribut = new String []{"0","1"} ; 
      
        for (int i = 0; i < this.dim.getValue(); i++) {
            attributes.addElement(new Attribute("nominal" + (i + 1),
                    Arrays.asList(attribut)));
        }
            this.attributes.addElement(sparseInstance.classAttribute());
            this.streamHeader = new InstancesHeader(new Instances(
                    getCLICreationString(InstanceStream.class), this.attributes, 0));
            this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        }


        int [] hashVal = hashVector(sparseInstance,this.dim.getValue(), Hashing.murmur3_128());
       
        return new InstanceExample(transformedInstance(sparseInstance, Arrays.stream(hashVal).asDoubleStream().toArray()));
    }



    public DenseInstance transformedInstance(Instance sparseInst, double [] hashVal) {

        Instances header = this.streamHeader;
        double[] attributeValues = new double[header.numAttributes()];

        for(int i = 0 ; i < header.numAttributes()-1 ; i++) {
            attributeValues[i] = hashVal[i];
        }

        attributeValues[attributeValues.length-1] = sparseInst.classValue();
        DenseInstance newInstance = new DenseInstance(1.0, attributeValues);
        newInstance.setDataset(header);
        header.add(newInstance);
        return newInstance;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO Auto-generated method stub
    }


    public  int[] hashVector(Instance instance, int n,
                                com.google.common.hash.HashFunction hashFunction) {

        int [] denseValues = new int [n];
        for (int i = 0 ; i < n ; i++) {
            denseValues[i] = 0;
        }
        for (int i = 0; i < instance.numAttributes()-1 ; i++){
                double diff = Math.abs(instance.value(i));
                if( diff  > Double.MIN_NORMAL) {
                    int  hash = hashFunction.hashInt(i).asInt();
                    int bucket = Math.abs(hash) % n;
                    denseValues[bucket] = (1);
                }
            }
        
        return denseValues;
    }


}