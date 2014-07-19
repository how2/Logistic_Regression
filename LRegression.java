package learning_hamr;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.regex.Pattern;

import org.kohsuke.args4j.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.etinternational.hamr.HAMR;
import com.etinternational.hamr.core.*;
import com.etinternational.hamr.resource.*;
import com.etinternational.hamr.resource.file.*;
import com.etinternational.hamr.resource.hdfs.*;
import com.etinternational.hamr.store.list.ArrayListContainer;
import com.etinternational.hamr.store.list.ValueList;
import com.etinternational.hamr.store.list.ValueListContainer;
import com.etinternational.hamr.store.list.ValueListStore;
import com.etinternational.hamr.store.number.NumberStore;
import com.etinternational.hamr.transform.Transform;

public class LRegression {
	private static final Logger LOG = LoggerFactory.getLogger(LRegression.class);
    private static final String HADOOP_HOME = System.getenv("HADOOP_HOME");
    private static final String HDFS_SITE_CONFIG = "file:///" + HADOOP_HOME + "/etc/hadoop/core-site.xml";
//    private static final String KV_OUTPUT_FORMAT = "%1$s : %2$s\n"; // Value : Key
    private static enum FileSystemType {
        HDFS,
        POSIX
    };

    //
    // Optional command line arguments
    //

    @Option(name="-h",usage="print this message",help=true,aliases={"--help"})
    private boolean displayUsage = false;

    @Option(name="-t",usage="file system type, default=HDFS",aliases={"--fileSystemType="})
    private FileSystemType fileSystem = FileSystemType.HDFS;

    @Option(name="-z",usage="value of $HADOOP_HOME",aliases={"--hadoopHome"},metaVar="STRING")
    private String hadoopHome = HADOOP_HOME;

    @Option(name="-n",usage="number output files, default=1",aliases={"--numOutputFiles="},metaVar="NUMBER")
    private int outputPartitions = 2;

    @Option(name="-i",usage="iteration times, default=1",aliases={"--numInterations="},metaVar="NUMBER")
    private int iterationTimes = 1;
    
    //
    // Required command line arguments
    //

//    @Option(name="-o",usage="output file name (REQUIRED)",required=true,aliases={"--outputFile="},metaVar="FILE")
//    private String outputFile = null;

    @Argument(usage="input file,directory,glob (one or more)",required=true,metaVar="FILE/DIRECTORY/GLOB")
    private final ArrayList<String> inputFiles = new ArrayList<>();
    
    // 
    private static final int D = 4;   // Number of dimensions
    private static final Random rand = new Random(42);
    private static final Pattern RE = Pattern.compile(","); 
    private static double[] weights = new double[D];
    
    public static int iter = 0;
    
    void run() {
        org.apache.hadoop.conf.Configuration hdfsConfig = null;
        ResourceReader<Long, String> reader = null;
//        ResourceWriter<Long, String> writer = null;
        
        if (fileSystem == FileSystemType.HDFS) {
            hdfsConfig = new org.apache.hadoop.conf.Configuration();
            hdfsConfig.addResource(new org.apache.hadoop.fs.Path(getHdfsSiteConfig()));
            LOG.info("Configuration: {}", getHdfsSiteConfig());
        }

        try {
        	
        	// Init
        	HAMR hamr = HAMR.initialize();
        
        	hamr.getSerializationRegistry().register(new GradientSerialization());
        	
        	// 
        	Job job = null;
        	
            Domain inputDomain = null;
            Domain computeDomain = null;
            Domain outputDomain = null;
            Domain broadcastDomain = null;
            
            // Notice: ArrayListContainer extends from ValueListContainer
            ValueListContainer<Long, Double> dataPoints = new ArrayListContainer<Double>();
            ValueListStore<Long, Double> arrayList = null;
            NumberStore<Integer, Double> vectorSum = null;          
 	
        	// Iteration
            for (iter = 1; iter <= iterationTimes; iter++) {

	        	job = new Job("LogisticRegression");
	        	
	            inputDomain = new Domain(job);
	            computeDomain = new Domain(job);
	            outputDomain = new Domain(job);
	            broadcastDomain = new Domain(job);
	        	
	        	// Set Partition num
	        	job.setPartitionCount(outputPartitions);
	
	            // Create the Resource Reader.
	            if (fileSystem == FileSystemType.HDFS) {
	                reader = new HdfsResourceReader<>(inputDomain, hdfsConfig, inputFiles.toArray(new String[0]));
	            }
	            else {
	                reader = new FileResourceReader<Long, String>(inputDomain, inputFiles.toArray(new String[0]));
	            }
	            
	            //reader.setConcurrentTaskLimit(1);
	            
	            // Create the Resource Writer.
	            // The format for converting the key/value pair into a String is specified.
//	            if (fileSystem == FileSystemType.HDFS) {
//	                writer = new HdfsResourceWriter<Long, String>(outputDomain, hdfsConfig, outputFile, KV_OUTPUT_FORMAT);
//	            } else {
//	                writer = new FileResourceWriter<Long, String>(outputDomain, outputFile, KV_OUTPUT_FORMAT);
//	            }

	        	// Init ArrayListStore, save m*n matrix	            
	            arrayList = new ValueListStore<Long, Double>(computeDomain, dataPoints);
	            
	            // Transform, read data and parse
	        	final Transform<Long, String, Long, Double> parsePoints = new Transform<Long, String, Long, Double>(inputDomain) {
	        		@Override
	        		public void apply(Long key, String value, PartitionConnector context) throws IOException {
	        			String[] tok = RE.split(value);
	        			
	                    for (int i = 0; i < D + 1; i++) {	                      
	                      context.push(out(), key, Double.parseDouble(tok[i]));
	                    }
	        		}
	        	};
	            
	        	// Compute the gradient
	        	// notice: ValueListStore -> ValueList<Double>
	        	final Transform<Long, ValueList<Double>, Integer, Double> computeGradient = new Transform<Long, ValueList<Double>, Integer, Double>(computeDomain) {
	        		@Override
	        		public void apply(Long key, ValueList<Double> value, PartitionConnector context) throws IOException {
	        			double[] x = new double[D];
	        			for (int i = 0; i < D; i++) {
	        				x[i] = value.get(i);
	        			}
	        			double y = value.get(D);
	        			
	        			for (int i = 0; i < D; i++) {
	        				
	            	        double dot = dot(weights, x);
	            	        double g = (1 / (1 + Math.exp(-y * dot)) - 1) * y * x[i];
	            	        context.push(out(), i, g);
	            	    }
	        		}
	        	};
	        	

	        	// Use NumberStore to sum gradient, the only way to sum?
	        	vectorSum = new NumberStore<Integer, Double>(outputDomain, Double.class);
	        	
	        	// broadcast and update w, works in single node mode only
	        	final Transform<Integer, Double, Integer, Gradient> updateWeights = new Transform<Integer, Double, Integer, Gradient>(outputDomain) {
	        		@Override
	        		public void apply(Integer key, Double value, PartitionConnector context) throws IOException {   					
    					weights[key] = weights[key] - value;
    					
    					for (int i = 0; i < HAMR.getInstance().getNumHosts(); i++) {
    						if (i != HAMR.getInstance().getHostID()) {
    							context.push(out(), i, new Gradient(key, weights[key]));
    						}
    					}
	        		}
	        	};
	 	        
	        	// update locally
	        	final Transform<Integer, Gradient, Integer, Integer> updateLocalWeights = new Transform<Integer, Gradient, Integer, Integer>(broadcastDomain) {
	        		@Override
	        		public void apply(Integer key, Gradient value, PartitionConnector context) throws IOException {   					
    					
	        			weights[value.col_index] = value.col_value;
	        		}
	        	};
	        	
	        	// read lines from file and parse data to arrayList
	        	// only once
		        if (iter == 1) {
	        		reader.out().bind(parsePoints.in());	        	
	        		parsePoints.out().bind(arrayList.add());
	        	}
	        	
	        	// compute gradient and update weights
	        	// every time
	        	arrayList.list().bind(computeGradient.in());
	        	computeGradient.out().bind(vectorSum.sum());
	        	vectorSum.out().bind(updateWeights.in());
	        	
	        	updateWeights.out().bind(updateLocalWeights.in());
	        	     	
	        	// run
	        	hamr.runJob(job);
	        	
	        	// output 
	        	System.out.println("On iteration " + iter);
	        	System.out.print("w: ");
	        	printWeights(weights);
        	}        	
        }
        catch (Exception e) {
            HAMR.abort(e);
        }
        finally {
            HAMR.shutdown();
        }
    }

    /**
     * Parses the command line and then runs the example.
     *  
     * @param args command line
     */
    public static void main(String[] args) {
    	    	
        LRegression example = new LRegression();
        CmdLineParser parser = new CmdLineParser(example);
        parser.setUsageWidth(80);
        
        // Initialize w to a random value
        for (int i = 0; i < D; i++) {
          weights[i] = 2 * rand.nextDouble() - 1;
        }        
        System.out.print("Initial w: ");
        printWeights(weights);

        try {

            parser.parseArgument(args);

            if (example.displayUsage) {
                example.printUsage(parser, System.out, null);
                System.exit(0);
            }
            if (example.fileSystem == FileSystemType.HDFS && example.hadoopHome == null) {
                String error = "'HADOOP_HOME' not found in environment, use -z (--hadoopHome=)";
                example.printUsage(parser, System.err, error);
                System.exit(1);
            }
            
            long startTime=System.nanoTime(); 
            
            example.run();
            
            long endTime=System.nanoTime(); 
            System.out.println("time: "+(endTime-startTime)+"ns");

        } catch (CmdLineException e) {
            example.printUsage(parser, System.err, "ERROR: " + e.getMessage());
            System.exit(1);
        }
        
    }


    /**
     * Print usage and optional message to specified stream.
     * 
     * @param parser instance of command line parser
     * @param stream output stream to use (system out or err)
     */
    void printUsage(CmdLineParser parser, PrintStream stream, String message) {
        if (message != null) {
           stream.println(message);
        }
        stream.println(this.getClass().getSimpleName() + " [options...] arguments...");
        parser.printUsage(stream);
        stream.println();
    }

    
    /**
     * Returns the HDFS site configuration file using HADOOP_HOME from
     * either the environment or the command line.
     * 
     * @return the Hadoop core-site xml file
     */
    String getHdfsSiteConfig() {
        
        if (HADOOP_HOME == null || !HADOOP_HOME.equals(hadoopHome)) {
            return "file://" + hadoopHome + "/etc/hadoop/core-site.xml";
        }
        else {
            return HDFS_SITE_CONFIG;
        }
        
    }
    /**
     * Returns the dot of two arrays
     * @param a array a
     * @param b array b
     * @return double x
     */
    public static double dot(double[] a, double[] b) {
        double x = 0;
        for (int i = 0; i < D; i++) {
            x += a[i] * b[i];
        }
        return x;
    }

    /**
     * Prints array
     * @param a array
     */
    public static void printWeights(double[] a) {
        System.out.println(Arrays.toString(a));
    }
	

}

