import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.*;
import weka.classifiers.functions.*;
import weka.classifiers.lazy.*;
import mulan.transformations.PT6Transformation;
import mulan.data.MultiLabelInstances;	
import java.io.File;
import weka.core.Utils;
import java.io.*;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
//import com.google.common.collect.*;

/* Multi-label CP with x-fold cross-validation (Transductive) */

class Main
{
	public static void main(String[] args)
	{
		try{
	
	    	/* Get multi-label data set and transform it to n binary classification datasets */	
	    	String arffFilename = Utils.getOption("arff", args); 
	    	String xmlFilename = Utils.getOption("xml", args);
	    	String trFilename = "_ClassRed_"+arffFilename;
	    	String Class = Utils.getOption("class",args);
	    	int ClassInt = Integer.parseInt(Class);
	    	
	        MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);
	        int numberOfInstances = dataset.getNumInstances(); 
			PT6Transformation pt6 = new PT6Transformation();
			Instances data = pt6.transformInstances(dataset);
			data.setClassIndex(data.numAttributes() - 2);
			
			
			int numClasses = data.numClasses();
			Instances[] kdata = new Instances[numClasses];
			
			for(int j=0;j<numClasses;j++)
			{
				kdata[j] = new Instances(data,0);
				
			}
			
			//System.out.printf("%d\n",numClasses);
			
			//final transformation 
			for(int i=0;i<data.numInstances();i++)
			{
				Instance inst = data.instance(i);
				int label = (int) inst.classValue();
				
				kdata[label].add(inst);
				//System.out.printf("%f\n",inst.classValue());
				
			}
			double[][][] pvalues = new double[numClasses][numberOfInstances][2];
			
			//for each binary-class dataset
			for(int j=ClassInt;j<=ClassInt;j++)
			{
				System.out.printf("Dataset for class %d\n",j);
				ArffSaver saver = new ArffSaver();
				kdata[j].setClassIndex(data.numAttributes() - 1);
				//remove actual label
				kdata[j].deleteAttributeAt(data.numAttributes() - 2);
				
				AttributeSelection atr = new AttributeSelection();
				//ChiSquaredAttributeEval chisqrt = new ChiSquaredAttributeEval();
				CfsSubsetEval cfs = new CfsSubsetEval();

        		//Ranker rnk = new Ranker();
				//rnk.setNumToSelect(30);
				//atr.setSearch(rnk);
				atr.setEvaluator(cfs);
			
				atr.SelectAttributes(kdata[j]);
			
				Instances red_data = atr.reduceDimensionality(kdata[j]);
				System.out.printf("%d\n",red_data.numAttributes());
				//int attrs = kdata[j].numAttributes();
				
				//for(int d=0;d<attrs-1;d++)
	    		//	kdata[j].deleteAttributeAt(0);
	    		saver.setInstances(red_data);
	    		String ntrFilename = j+trFilename;
	    		saver.setFile(new File(ntrFilename));
	    		
	    		saver.writeBatch();
	    	
	    		
				int FOLDS = 10;
				int error = 0;
				int[] uncertainty = new int[31];
				int[] errorv = new int[31];
				
				for(int i=0;i<31;i++)
				{
					uncertainty[i] = 0;
					errorv[i] = 0;
				}
				int x=0;
				//start x-fold
		    	for(int fold=0;fold<FOLDS;fold++)
				{
					Instances train_set = red_data.trainCV(FOLDS,fold);
					Instances test_set = red_data.testCV(FOLDS,fold);
				
					int trainx = train_set.numInstances();
					int testx = test_set.numInstances();
					
					
					//test examples
					for(int t=0;t<testx;t++)
					{
						Instance test_example = test_set.instance(t);
						int actual = (int)test_example.classValue();
						
						
						CP cp = new CP();
						MultilayerPerceptron ann = new MultilayerPerceptron();
						//IBk knn = new IBk(20);
						//ann.setTrainingTime(200);
						ann.setValidationSetSize(10);
						cp.setClassifier(ann);
						
						//for each label of test example
						for(int c=0;c<2;c++)
						{
							//change the label of test example					
							test_example.setClassValue((double) c);

							//and add it to the training set
							train_set.add(test_example);
							
							cp.buildConformalPredictor(train_set);
							pvalues[j][x][c] = cp.getPvalue();
							
							//delete test example from trainset
							train_set.delete(trainx); 
						}
						x++;
						int prediction = (int) cp.classifyInstance(pvalues[j][x-1]);
						double confidence = cp.getConfidence(pvalues[j][x-1]);
						double credibility = cp.getCredibility(pvalues[j][x-1]);
						
						for(int s=0;s<=30;s++)
						{
							double[] Rk = cp.getRegion((100-s)*0.01,pvalues[j][x-1]);
							if(Rk.length > 1)
								uncertainty[s]++;
							
							errorv[s] += errorf(s*0.01,pvalues[j][x-1],actual,2);
								
						}
						
						System.out.printf("%d:%d: Prediction: %d, Actual: %d, Confidence: %.4f, Credibility: %.4f\n",fold,t,prediction,actual,confidence,credibility);
						
						if(actual != prediction)
							error++;
							 
					} //end of test examples
					
				}	//end of folds
		    		
		    		
				System.out.printf("Accuracy: %.4f\n",1-((double)error/(double) red_data.numInstances()));
				for(int i=0;i<31;i++)
				{
					System.out.printf("Certainty at %d%% confidence: %.4f - Error: %.4f\n",100-i,1 - ((double) uncertainty[i]/ (double) red_data.numInstances()), (double) errorv[i]/ (double) red_data.numInstances());
				}
		
			} //end of datasets
			
			
			
			for(int j=ClassInt;j<=ClassInt;j++)
			{
				String filename = "pvalues"+j+".txt";
				FileWriter fstream = new FileWriter(filename);
        		BufferedWriter out = new BufferedWriter(fstream);
				for(int x=0;x<numberOfInstances;x++)
				{
					for(int c=0;c<2;c++)
					{
						String aString = Double.toString(pvalues[j][x][c]);
						out.write(aString);
						if(c==0)
							out.write(" ");	
					}
					out.write("\n");
				}
				out.close();
			}
			
			
			
		} //end of try
		catch(Exception e)
		{
			e.printStackTrace();
		}
	} //end main
	
	static int errorf(double err,double[] p,int actual,int numClasses)
	{
		int y=1;
		for(int i=1; i<=numClasses;i++)
	    {
	
	    	if(p[i-1] > err && i-1 == actual)
	     	{
	     		y = 0;
	        	break;
	    	}
		}
		return y;
	}
	
}
