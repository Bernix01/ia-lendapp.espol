	�V-�?�V-�?!�V-�?	�"	�?J@�"	�?J@!�"	�?J@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�V-�?�A`��"�?AZd;�O��?Y�v��/�?*	     ��@2O
Iterator::Model::BatchV2��x�&1�?!u�1��X@)�/�$�?1��ڈu�T@:Preprocessing2X
!Iterator::Model::BatchV2::Shuffle ��MbX�?!����!/@)��MbX�?1����!/@:Preprocessing2F
Iterator::Model'1�Z�?!      Y@){�G�zt?1<Eg@(�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 52.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2s8.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�"	�?J@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�A`��"�?�A`��"�?!�A`��"�?      ��!       "      ��!       *      ��!       2	Zd;�O��?Zd;�O��?!Zd;�O��?:      ��!       B      ��!       J	�v��/�?�v��/�?!�v��/�?R      ��!       Z	�v��/�?�v��/�?!�v��/�?JCPU_ONLYY�"	�?J@b 