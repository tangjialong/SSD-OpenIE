import numpy as np

def softmax(z):
	return np.exp(z)/sum(np.exp(z))

input_filepath1 = 'INPUT_FILEPATH'
input_filepath2 = 'data/tmp/pred.tsv' # score filepath
output_filepath = 'OUTPUT_FILEPATH'

with open(input_filepath1) as input_file1:
	with open(input_filepath2) as input_file2:
		with open(output_filepath, 'w') as output_file:
			for line1, line2 in zip(input_file1, input_file2):
				line1 = line1.strip().split('\t')
				line2 = line2.strip().split('\t')
				tmp = np.array([float(line2[4]), float(line2[5])])
				tmp = softmax(tmp)
				# rerank_score = str((float(line1[1]))) # AVG log
				# rerank_score = str((np.log(tmp[1]))) # Semantic Consistency
				rerank_score = str((np.log(tmp[1]) + float(line1[1]))) # AVG log + Semantic Consistency
				output_file.write(line1[0] + '\t' + rerank_score + '\t' + '\t'.join(line1[2:]) + '\n')