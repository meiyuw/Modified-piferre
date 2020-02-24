# Modified-piferre
A modified piferre script that is compatible with the desi MWS pipeline integration script.

Currently the differences between this script and the original piferre script (https://github.com/callendeprieto/piferre) are:

(1) The users can specify which spectra-64 fit file to be processed rather than process the whol spectra-64 folder instead.

(2) The users can specify where to dump the output files (--output_dir) and all the scripts and parameter files (--output_script_dir)

(3) It can process only MWS targets and also those fibers within the desired exposure ID range.

(4) If the same spectra-64 file has been processed before and the corresponding sptab or spmod files have been generated, the new results will be merged with the old output files.

     python piferre.py [--input_files INPUT_FILES] [--input_dir INPUT_DIR]

                 [--output_dir OUTPUT_DIR] [--output_script_dir OUTPUT_SCRIPT_DIR] 
                 
                 [--minexpid MINEXPID] [--maxexpid MAXEXPID] [--allobjects]

**optional arguments:**

    --input_files         Read the list of spectral files from a text file.
  
    --input_dir           Directory of input files to be processed.
  
    --output_dir          Directory for the output files.
  
    --output_script_dir   Output directory for the slurm scripts and ferre input files.
    
    --allobjects          Process not just MWS targets but all target types.
    
    --minexpid            Minimun expid value.
    
    --maxexpid            Max expid value.
  
                       
