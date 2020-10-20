input_filepath = 'INPUT_FILEPATH'
output_filepath = 'data/tmp/test.tsv' # score filepath

with open(input_filepath) as input_file:
	with open(output_filepath, 'w') as output_file:
		for line in input_file:
			line = line.strip().split('\t')
			sentence = line[0]
			args = line[2:]
			args[0] = '@@ ' + args[0]
			sorted_args = []
			for arg in  args:
				sorted_args.append(arg.split('##'))
			sorted_args = sorted(sorted_args, key=lambda x:x[1])
			sorted_args = [s_a[0] for s_a in sorted_args]
			sorted_args = ' '.join(sorted_args)
			output_file.write(sentence + '\t' + sorted_args + '\t0\n')