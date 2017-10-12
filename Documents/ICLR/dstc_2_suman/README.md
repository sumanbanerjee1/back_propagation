# dstc_2_suman

The pre-processing creates three pickle files (one each for train,test,dev)
The files are list datastructure consisting of five lists - pre_kb,kb,post_kb,utterance,response
Items in the list are word tokens with &lt;eos&gt; for normal sentences and &lt;eok&gt;	for individual kb enteries
&lt;beg&gt; marks the beggining of dialog.
Give source directory and target directory to start the process
