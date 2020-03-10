#!/home/callende/anaconda3/bin/python3
'''
Interface to use FERRE from python for DESI/BOSS/WEAVE data

use: piferre path-to-spectra [truthfile]

e.g. piferre 
/home/callende/work/desi/berkeley/spectro/redux/dc17a2/spectra-64 

Author C. Allende Prieto
'''
import pdb
import sys
import stat
import os
import glob
import re
import astropy
import argparse
from numpy import arange,loadtxt,savetxt,zeros,ones,nan,sqrt,interp,concatenate,array,reshape,min,max,where,divide,mean, stack
import numpy as np
from astropy.io import fits
import astropy.table as tbl
import astropy.units as units
import matplotlib.pyplot as plt
import subprocess
import datetime, time
from datetime import date

#extract the header of a synthfile
def head_synth(synthfile):
    file=open(synthfile,'r')
    line=file.readline()
    header={}
    while (1):
        line=file.readline()
        part=line.split('=')
        if (len(part) < 2): break
        k=part[0].strip()
        v=part[1].strip()
        header[k]=v
    return header

#extract the wavelength array for a FERRE synth file
def lambda_synth(synthfile):
    header=head_synth(synthfile)
    tmp=header['WAVE'].split()
    npix=int(header['NPIX'])
    step=float(tmp[1])
    x0=float(tmp[0])
    x=arange(npix)*step+x0
    if header['LOGW']:
      if int(header['LOGW']) == 1: x=10.**x
      if int(header['LOGW']) == 2: x=exp(x)  
    return x

#create a slurm script for a given pixel
def write_slurm(sdir,pixel,output_path,n_fiber,nthreads=1,script_path=None,ngrids=None, suffix=''):
    ferre=os.environ['DESI_MWS_root']+"/ferre/src/a.out"
    python_path=os.environ['DESI_MWS_root']+"/piferre"
    try:   
      host=os.environ['HOST']
    except:
      host='Unknown'

    now=time.strftime("%c")
    if script_path is None: path='.'
    if ngrids is None: ngrids=1

    # Calculating the required runtime. For 300 fibers it takes about 20 mins.
    # Each run request at least 20 min.
    runtime=20.0*60.0*np.ceil(n_fiber/300.0) # in second.
    rt_5m=np.ceil(runtime/(5.0*60.0))
    rt_hr=int(rt_5m//12)
    if(rt_hr != 0):
        rt_min=int((rt_5m%12)*5)
    else:
        if ((rt_5m%12)*5 > 20.0):
            rt_min=int((rt_5m%12)*5)
        else:
            rt_min=20

    f=open(os.path.join(script_path,sdir,pixel,pixel+suffix+'.slurm'),'w')
    f.write("#!/bin/bash \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("#This script was written by piferre.py on "+now+" \n") 
    if host[:4] == 'cori':
      f.write("#SBATCH --qos=regular" + "\n")
      f.write("#SBATCH  -o "+str(pixel)+"_%j.out"+" \n")
      f.write("#SBATCH  -e "+str(pixel)+"_%j.err"+" \n")      
      f.write("#SBATCH --constraint=haswell" + "\n") 
      f.write("#SBATCH --time="+str(rt_hr)+":"+str(100+rt_min)[1:]+":00 \n")
      f.write("#SBATCH --ntasks=1" + "\n")
      f.write("#SBATCH --cpus-per-task="+str(nthreads*2)+"\n")
    else:
      f.write("#SBATCH  -J "+str(pixel)+" \n")
      f.write("#SBATCH  -p long \n")
      f.write("#SBATCH  -o "+str(pixel)+"_%j.out"+" \n")
      f.write("#SBATCH  -e "+str(pixel)+"_%j.err"+" \n")
      f.write("#SBATCH  -n "+str(nthreads)+" \n")
      f.write("#SBATCH  -t 00:10:00"+" \n") #hh:mm:ss
      f.write("#SBATCH  -D "+os.path.join(script_path,sdir,pixel)+" \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("export OMP_NUM_THREADS="+str(nthreads)+"\n")
    f.write("source "+os.environ['DESI_MWS_root']+"/setup.sh \n")
    f.write("cd "+os.path.join(script_path,sdir,pixel)+"\n")
    for i in range(ngrids): 
      #f.write("cp input.nml"+suffix+"_"+str(i)+" input.nml \n")
      f.write("time "+ferre+" "+os.path.join(script_path,sdir,pixel)+"/input.nml"+suffix+"_"+str(i)+" >& log_"+str(i))
      if (i == 8): 
        f.write( "  \n")
      else:
        f.write( " & \n") 
    if ngrids > 1:
      f.write("python3 -c \"import sys; sys.path.insert(0, '"+python_path+ \
              "'); from piferre import opfmerge, write_tab_fits, write_mod_fits; opfmerge(\'"+\
              str(pixel)+suffix+"\', path='"+os.path.join(script_path,sdir,pixel)+"',wait_on_sorted=True); write_tab_fits(\'"+\
              str(pixel)+suffix+"\', '"+os.path.join(script_path,sdir,pixel)+"', path='"+os.path.join(output_path,sdir,pixel)+"'); write_mod_fits(\'"+\
              str(pixel)+suffix+"\', '"+os.path.join(script_path,sdir,pixel)+"', path='"+os.path.join(output_path,sdir,pixel)+"')\"\n")              
    f.close()
    os.chmod(os.path.join(script_path,sdir,pixel,pixel+suffix+'.slurm'),0o777)

    return None


  
#create a FERRE control hash (content for a ferre input.nml file)
def mknml(synthfiles,root,k,order,path=None,nthreads=1):
    if path is None: path="."
    path=os.path.abspath(path)
    header=head_synth(synthfiles[0])
    nml={}
    ndim=int(header['N_OF_DIM'])
    nml['NDIM']=ndim
    nml['NOV']=ndim
    nml['INDV']=' '.join(map(str,arange(ndim)+1))
    for i in range(len(synthfiles)): nml['SYNTHFILE('+str(i+1)+')'] = "'"+os.path.join(path,synthfiles[i])+"'"
    nml['PFILE'] = "'"+root+".vrd"+"'"
    nml['FFILE'] = "'"+root+".frd"+"'"
    nml['ERFILE'] = "'"+root+".err"+"'"
    nml['OPFILE'] = "'"+root+".opf"+str(k)+"'"
    nml['OFFILE'] = "'"+root+".mdl"+str(k)+"'"
    nml['SFFILE'] = "'"+root+".nrd"+str(k)+"'"
    #nml['WFILE'] = "'"+root+".wav"+"'"
    nml['ERRBAR']=1
    nml['COVPRINT']=1
    #nml['WINTER']=2
    nml['INTER']=order
    nml['ALGOR']=3
    #nml['GEN_NUM']=5000
    #nml['NRUNS']=2**ndim
    #nml['INDINI']=''
    #for i in range(ndim): nml['INDINI']=nml['INDINI']+' 2 '
    nml['NTHREADS']=nthreads
    nml['F_FORMAT']=0
    nml['F_ACCESS']=0
    #nml['CONT']=1
    #nml['NCONT']=0
    nml['CONT']=3
    nml['NCONT']=50

    return nml

#write out a FERRE control hash to an input.nml file
def writenml(nml,nmlfile='input.nml',path=None):
    if path is None: path='./'
    f=open(os.path.join(path,nmlfile),'w')
    f.write('&LISTA\n')
    for item in nml.keys():
        f.write(str(item))
        f.write("=")
        f.write(str(nml[item]))
        f.write("\n")
    f.write(" /\n")
    f.close()
    return None

#run FERRE
def ferrerun(path=None):
    if path is None: path="./"
    pwd=os.path.abspath(os.curdir)
    os.chdir(path)
    ferre="/home/callende/ferre/src/a.out"
    code = subprocess.call(ferre)
    os.chdir(pwd)
    return code


#read redshift derived by the DESI pipeline
def readzbest(filename):
  hdu=fits.open(filename)
  if len(hdu) > 1:
    enames=extnames(hdu)
    #print(enames)
    if 'ZBEST' in enames:
      zbest=hdu['zbest'].data
      targetid=zbest['targetid'] #array of long integers
    else:
      zbest=hdu[1].data
      plate=zbest['plate']
      mjd=zbest['mjd']
      fiberid=zbest['fiberid']
      targetid=[]
      for i in range(len(plate)): 
        targetid.append(str(plate[i])+'-'+str(mjd[i])+'-'+str(fiberid[i]))
      targetid=array(targetid)  #array of strings

    #print(type(targetid),type(zbest['z']),type(targetid[0]))
    #print(targetid.shape,zbest['z'].shape)
    z=dict(zip(targetid, zbest['z']))
  else:
    z=dict()

  return(z)

#read redshift derived by the Koposov pipeline
def readk(filename):
  clight=299792.458 #km/s
  hdu=fits.open(filename)
  if len(hdu) > 1:
    k=hdu[1].data
    #targetid=k['targetid']
    targetid=k['fiber']
    #teff=k['teff']
    #logg=k['loog']
    #vsini=k['vsini']
    #feh=k['feh']
    #z=k['vrad']/clight
    z=dict(zip(targetid, k['vrad']/clight))  
    #z_err=dict(zip(k['target_id'], k['vrad_err']/clight))  
  else:
    z=dict() 
  return(z)

#read truth tables (for simulations)
def readtruth(filename):
  hdu=fits.open(filename)
  truth=hdu[1].data
  targetid=truth['targetid']
  feh=dict(zip(targetid, truth['feh']))
  teff=dict(zip(targetid, truth['teff']))
  logg=dict(zip(targetid, truth['logg']))
  #rmag=dict(zip(targetid, truth['flux_r']))
  rmag=dict(zip(targetid, truth['mag']))
  z=dict(zip(targetid, truth['truez']))
  return(feh,teff,logg,rmag,z)

#read spectra
def readspec(filename,band=None):

  hdu=fits.open(filename)

  if filename.find('spectra-64') > -1 or filename.find('exp_') > -1: #DESI
    wavelength=hdu[band+'_WAVELENGTH'].data #wavelength array
    flux=hdu[band+'_FLUX'].data       #flux array (multiple spectra)
    ivar=hdu[band+'_IVAR'].data       #inverse variance (multiple spectra)
    #mask=hdu[band+'_MASK'].data       #mask (multiple spectra)
    res=hdu[band+'_RESOLUTION'].data  #resolution matrix (multiple spectra)
    #bintable=hdu['BINTABLE'].data  #bintable with info (incl. mag, ra_obs, dec_obs)

  if filename.find('spPlate') > -1: #SDSS/BOSS
    header=hdu['PRIMARY'].header
    wavelength=header['CRVAL1']+arange(header['NAXIS1'])*header['CD1_1'] 
#wavelength array
    wavelength=10.**wavelength
    flux=hdu['PRIMARY'].data       #flux array (multiple spectra)
    #ivar=hdu['IVAR'].data       #inverse variance (multiple spectra)    
    ivar=hdu[1].data       #inverse variance (multiple spectra)
    #andmask=hdu['ANDMASK'].data       #AND mask (multiple spectra)  
    #ormask=hdu['ORMASK'].data       #OR mask (multiple spectra)
    #res=hdu['WAVEDISP'].data  #FWHM array (multiple spectra)
    res=hdu[4].data  #FWHM array (multiple spectra)
    #bintable=hdu['BINTABLE'].data  #bintable with info (incl. mag, ra, dec)
    

  return((wavelength,flux,ivar,res))

#write piferre param. output
def write_tab_fits(pixel, script_path,path=None):
  
  if path is None: path=""
  root=os.path.join(script_path,pixel)
  #print('write_tab_fit root=',root) 
  o=glob.glob(root+".opf")
  m=glob.glob(root+".mdl")
  n=glob.glob(root+".nrd")
  fmp=glob.glob(root+".fmp.fits")
  
  success=[]
  fid=[]
  teff=[]
  logg=[]
  feh=[]
  alphafe=[]
  micro=[]
  covar=[]
  elem=[]
  elem_err=[]
  snr_med=[]
  chisq_tot=[]
  of=open(o[0],'r')
  for line in of:
    cells=line.split()
    if (len(cells) == 19):
      #Kurucz grids with 3 dimensions: id, 3 par, 3 err, 0., med_snr, lchi, 9 cov, 
      if (float(cells[9]) < 1. and float(cells[8]) > 5.): 
        success.append(1) 
      else: success.append(0)
      fid.append(cells[0])
      feh.append(float(cells[1]))
      teff.append(float(cells[2]))
      logg.append(float(cells[3]))
      alphafe.append(nan)
      micro.append(nan)
      elem.append([nan,nan])
      elem_err.append([nan,nan])
      chisq_tot.append(10.**float(cells[9]))
      snr_med.append(float(cells[8]))
      cov = reshape(array(cells[10:],dtype=float),(3,3))
      covar.append(cov)
    else:
      #white dwarfs 2 dimensions: id, 2 par, 2err, 0.,med_snr, lchi, 4 cov
      if (float(cells[7]) < 1. and float(cells[6]) > 5.): 
        success.append(1) 
      else: success.append(0)
      fid.append(cells[0])
      feh.append(-10.)
      teff.append(float(cells[1]))
      logg.append(float(cells[2]))
      alphafe.append(nan)
      micro.append(nan)
      elem.append([nan,nan])
      elem_err.append([nan,nan])
      chisq_tot.append(10.**float(cells[7]))
      snr_med.append(float(cells[6]))
      cov = zeros((3,3))
      cov[1:,1:] = reshape(array(cells[8:],dtype=float),(2,2))
      #cov = reshape(array(cells[8:],dtype=float),(2,2))
      covar.append(cov)    

  hdu0=fits.PrimaryHDU()
  now = datetime.datetime.fromtimestamp(time.time())
  nowstr = now.isoformat() 
  nowstr = nowstr[:nowstr.rfind('.')]
  hdu0.header['DATE'] = nowstr
  hdulist = [hdu0]

  cols = {}
  cols['success'] = success
  cols['fid'] = fid
  cols['teff'] = array(teff)*units.K
  cols['logg'] = array(logg)
  cols['feh'] = array(feh)
  cols['alphafe'] = array(alphafe) 
  cols['micro'] = array(micro)*units.km/units.s
  cols['covar'] = array(covar).reshape(len(success),3,3)
  cols['elem'] = array(elem)
  cols['elem_err'] = array(elem_err)
  cols['chisq_tot'] = array(chisq_tot)
  cols['snr_med'] = array(snr_med)

  colcomm = {
  'success': 'Bit indicating whether the code has likely produced useful results',
  'fid': 'Identifier used in FERRE to associate input/output files',
  'teff': 'Effective temperature',
  'logg': 'Surface gravity (g in cm/s**2)',
  'feh': 'Metallicity [Fe/H] = log10(N(Fe)/N(H)) - log10(N(Fe)/N(H))sun' ,
  'alphafe': 'Alpha-to-iron ratio [alpha/Fe]',
  'micro': 'Microturbulence',
  'covar': 'Covariance matrix for (Teff,logg,[Fe/H])',
  'elem': 'Elemental abundance ratios to iron [elem/Fe]',
  'elem_err': 'Uncertainties in the elemental abundance ratios to iron',
  'chisq_tot': 'Total chi**2',
  'snr_med': 'Median signal-to-ratio'
  }      

  
  table = tbl.Table(cols)
  hdu=fits.BinTableHDU(table,name = 'SPTAB')
  #hdu.header['EXTNAME']= ('SPTAB', 'Stellar Parameter Table')
  i = 0
  for entry in colcomm.keys():
    #print(entry) 
    hdu.header['TCOMM'+str(i+1)] = colcomm[entry]
    i+=1
  hdulist.append(hdu)


  if len(fmp) > 0:
    ff=fits.open(fmp[0])
    fibermap=ff[1]
    hdu=fits.BinTableHDU.from_columns(fibermap, name='FIBERMAP')
    #hdu.header['EXTNAME']='FIBERMAP'
    hdulist.append(hdu)

  hdul=fits.HDUList(hdulist)
  
  # Merge with the old sptab files or write a new file if no old files exist.
  sptab_name=path+'/sptab-64-'+pixel+'.fits'
  if not os.path.exists(sptab_name):
  	hdul.writeto(sptab_name)
  else:
  	merge_hdus(hdulist, sptab_name)
  	
  
  return None
    
#write piferre spec. output  
def write_mod_fits(pixel, script_path,path=None):  
  
  if path is None: path="" 
  root=os.path.join(script_path,pixel)
  
  xbandfiles = sorted(glob.glob(root+'-*.wav'))
  band = []
  npix = []
  for entry in xbandfiles:
    match = re.search('-[\w]*.wav',entry)
    tag = match.group()[1:-4]
    if match: band.append(tag.upper())
    x = loadtxt(root+'-'+tag+'.wav')
    npix.append(len(x))
    
  x = loadtxt(root+'.wav')
  if len(npix) == 0: npix.append(len(x))

  m=glob.glob(root+".mdl")
  e=glob.glob(root+".err")
  n=glob.glob(root+".nrd")

  fmp=glob.glob(root+".fmp.fits")  
  mdata=loadtxt(m[0])
  edata=loadtxt(e[0])  
  odata=loadtxt(n[0])
  if (len(n) > 0 & len(odata) > 0):     
    f=glob.glob(root+".frd")
    fdata=loadtxt(f[0])
    #print('data shape:',edata.shape,fdata.shape,odata.shape)
    edata=edata/fdata*odata
  else:
    odata=loadtxt(root+".frd")  

  hdu0=fits.PrimaryHDU()
  now = datetime.datetime.fromtimestamp(time.time())
  nowstr = now.isoformat() 
  nowstr = nowstr[:nowstr.rfind('.')]
  hdu0.header['DATE'] = nowstr
  hdulist = [hdu0]

  i = 0
  j1 = 0

  for entry in band:
    j2 = j1 + npix[i] 
    #print(entry,i,npix[i],j1,j2)
    #colx = fits.Column(name='wavelength',format='e8', array=array(x[j1:j2]))
    #coldefs = fits.ColDefs([colx])
    #hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu = fits.ImageHDU(name=entry+'_WAVELENGTH', data=x[j1:j2])
    #hdu.header['EXTNAME']=entry+'_WAVELENGTH'
    hdulist.append(hdu)
    
    if odata.ndim == 2: tdata = odata[:,j1:j2]
    else: tdata = odata[j1:j2][None,:]
    col01 = fits.Column(name='obs',format=str(npix[i])+'e8', dim='('+str(npix[i])+')', array=tdata)
    if edata.ndim == 2: tdata = edata[:,j1:j2]
    else: tdata = edata[j1:j2][None,:]
    col02 = fits.Column(name='err',format=str(npix[i])+'e8', dim='('+str(npix[i])+')', array=tdata)
    if mdata.ndim == 2: tdata = mdata[:,j1:j2]
    else: tdata = mdata[j1:j2][None,:]
    col03 = fits.Column(name='fit',format=str(npix[i])+'e8', dim='('+str(npix[i])+')', array=tdata)    
    coldefs = fits.ColDefs([col01,col02,col03])
    hdu=fits.BinTableHDU.from_columns(coldefs, name=entry+'_MODEL')
    #hdu = fits.ImageHDU(name=entry+'_MODEL', data=stack([odata[:,j1:j2],edata[:,j1:j2],mdata[:,j1:j2]]) ) 
    #hdu.header['EXTNAME']=entry+'_MODEL'
    hdulist.append(hdu)
    i += 1
    j1 = j2

  if len(fmp) > 0:
    ff=fits.open(fmp[0])
    fibermap=ff[1]
    hdu=fits.BinTableHDU.from_columns(fibermap, name='FIBERMAP')
    #hdu.header['EXTNAME']='FIBERMAP'
    hdulist.append(hdu)

  hdul=fits.HDUList(hdulist)
  
  spmod_name=path+'/spmod-64-'+pixel+'.fits'
  if not os.path.exists(spmod_name):
  	hdul.writeto(spmod_name)
  else:
  	merge_hdus(hdulist, spmod_name)  
  
  return None
  
def merge_hdus(hdus, ofile):
	# TODO: it doesn't deal with combined data

    for i in range(len(hdus)):
        if i == 0:
            continue
        curhdu = hdus[i]
        curhduname = curhdu.name

        if curhduname in ['FIBERMAP', 'SPTAB']:
            newdat = tbl.Table(curhdu.data)
            olddat = tbl.Table().read(ofile, hdu=curhduname)
            curouthdu = tbl.vstack((olddat, newdat))
            hdus[i] = curouthdu
            continue
        if curhduname[-10:] == 'WAVELENGTH':
            continue
        if curhduname[-5:] == 'MODEL':
            newdat = curhdu.data
            olddat = pyfits.getdata(ofile, curhduname)
            hdus[i] = pyfits.ImageHDU(np.concatenate(
                (olddat, newdat), axis=0),
                                      name=curhduname)
            continue
        raise Exception('This hdus "',curhdunam,'" should not be here')

    astropy.io.fits.HDUList(hdus).writeto(ofile, overwrite=True)  

#write ferre files
def write_ferre_input(out_script_path,root,ids,par,y,ey,path=None,suffix=''):

  if path is None: path="./"

  #open ferre input files
  vrd=open(os.path.join(out_script_path,path,root)+suffix+'.vrd','w')
  frd=open(os.path.join(out_script_path,path,root)+suffix+'.frd','w')
  err=open(os.path.join(out_script_path,path,root)+suffix+'.err','w')

  nspec, freq = y.shape

  #loop to write data files
  i=0
  while (i < nspec):

    #print(str(i)+' ')

    #vrd.write("target_"+str(i+1)+" 0.0 0.0 0.0")
    ppar=[ids[i]]
    for item in par[ids[i]]: ppar.append(item)
    #vrd.write(' '.join(map(str,ppar))
    vrd.write("%30s %6.2f %10.2f %6.2f %6.2f %12.9f %12.9f %12.9f %12.9f\n" % 
tuple(ppar) )    
    #ppar.tofile(ppar,sep=" ",format="%s")
    #vrd.write("\n")

    yy=y[i,:]
    yy.tofile(frd,sep=" ",format="%0.4e")
    frd.write("\n")
    eyy=ey[i,:]
    eyy.tofile(err,sep=" ",format="%0.4e")
    err.write("\n")
    i+=1
    #print(i,yy[0],eyy[0])
 
  #close files
  vrd.close()
  frd.close()
  err.close()


def opfmerge(pixel,path=None,wait_on_sorted=False):

  if path is None: path="./"
  root=os.path.join(path,pixel) 

  if wait_on_sorted:
    o=sorted(glob.glob(root+".opf*_sorted"))  
    while (len(o) > 0):
      time.sleep(5)
      o=sorted(glob.glob(root+".opf*_sorted"))
      

  o=sorted(glob.glob(root+".opf?"))
  m=sorted(glob.glob(root+".mdl?"))
  n=sorted(glob.glob(root+".nrd?"))
  
  llimit = [3500.,5500.,7000.,10000.,20000.,6000.,10000.,10000.,15000.]
  iteff = [2,     2,     2,    2,     2,      2,    2,     2,     2   ]
  ilchi = [9,     9,     9,    9,     9,      7,    7,     7,     7   ]

  ngrid=len(o)
  if ngrid != len(m): 
    print("there are different number of opf? and mdl? arrays")
    return(0)
  if (len(n) > 0):
    if ngrid != len(m):  
      print("there are different number of opf? and mdl? arrays")
      return(0)

  #open input files
  of=[]
  mf=[]
  if len(n) > 0: nf=[]
  for i in range(len(o)):
    of.append(open(o[i],'r'))
    mf.append(open(m[i],'r'))
    if len(n) > 0: nf.append(open(n[i],'r'))

  #open output files
  oo=open(root+'.opf','w')
  mo=open(root+'.mdl','w')
  if len(n) > 0: no=open(root+'.nrd','w')
  	
  for line in of[0]: 
    array=line.split()
    min_chi=float(array[ilchi[0]])
    min_oline=line
    #print(min_chi,min_oline)
    min_mline=mf[0].readline()
    if len(n) > 0: min_nline=nf[0].readline()
    for i in range(len(o)-1):
      oline=of[i+1].readline()
      mline=mf[i+1].readline()
      if len(n) > 0: nline=nf[i+1].readline()
      array=oline.split()
      #print(i,of[i+1])
      #print(array)
      #print(float(array[ilchi[i+1]]))
      if float(array[ilchi[i+1]]) < min_chi and float(array[iteff[i+1]]) > llimit[i+1]*1.01: 
        min_chi=float(array[ilchi[i+1]])
        min_oline=oline
        min_mline=mline
        if len(n) > 0: min_nline=nline
    
    #print(min_chi,min_oline)
    oo.write(min_oline)
    mo.write(min_mline)
    if len(n) > 0: no.write(min_nline)
  
  #close input files
  for i in range(len(o)):
    #print(o[i],m[i])
    of[i].close
    mf[i].close
    if len(n) > 0: nf[i].close

  #close output files
  oo.close
  mo.close
  if len(n) > 0: no.close
  
  return None

#get names of extensions from a FITS file
def extnames(hdu):
  #hdu must have been open as follows hdu=fits.open(filename)
  x=hdu.info(output=False)
  names=[]
  for entry in x: names.append(entry[1])
  return(names)

#identify input data files and associated zbest files 
def finddatafiles(path,pixel,sdir='',rvpath=None):
	if rvpath is None: rvpath = path
	
	infiles=os.listdir(os.path.join(path,sdir,pixel))
	datafiles=[]
	zbestfiles=[]
	for ff in infiles: #add subdirs, which may contain zbest files for SDSS/BOSS
		if os.path.isdir(os.path.join(path,sdir,pixel,ff)):
			extrafiles=os.listdir(os.path.join(path,sdir,pixel,ff))
			for ff2 in extrafiles:
				infiles.append(os.path.join(ff,ff2))
	
	if rvpath != path:
		if os.path.isdir(os.path.join(rvpath,sdir,pixel)):
			infiles2=os.listdir(os.path.join(rvpath,sdir,pixel))
		for ff in infiles2: #add subdirs, which may contain zbest files for SDSS/BOSS
			infiles.append(ff)
			if os.path.isdir(os.path.join(rvpath,sdir,pixel,ff)):
				extrafiles2=os.listdir(os.path.join(rvpath,sdir,pixel,ff))
				for ff2 in extrafiles2:
					infiles.append(os.path.join(ff,ff2))
	infiles.sort()
	#print('infiles=',infiles)
	
	for filename in infiles:
  		#print('filename=',filename)
  		# DESI sims/data
  		if (filename.find('spectra-64') > -1 and filename.find('.fits') > -1):
  			datafiles.append(os.path.join(path,sdir,pixel,filename))
  		if (filename.find('zbest-64') > -1 and filename.find('.fits') > -1):
  			zbestfiles.append(os.path.join(path,sdir,pixel,filename))
  		elif (filename.find('rvtab') > -1 and filename.find('.fits') > -1):
  			zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))

	#analyze the situation wrt input files
	ndatafiles=len(datafiles)
	nzbestfiles=len(zbestfiles)
	print ('Found '+str(ndatafiles)+' input data files')
	for filename in datafiles: print(filename+'--')
	print ('and '+str(nzbestfiles)+' associated zbest files')
	for filename in zbestfiles: print(filename+'--')
	
	if (ndatafiles != nzbestfiles):
		print('ERROR -- there is a mismatch between the number of data files and zbest files, this pixel is skipped')
		return (None,None)
	
	return (datafiles,zbestfiles)

#pack a collection of fits files with binary tables in multiple HDUs into a single one
def packfits(input="*.fits",output="output.fits"):


  f = glob.glob(input)

  print('reading ... ',f[0])
  hdul1 = fits.open(f[0])
  hdu0 = hdul1[0]
  for entry in f[1:]:       
    print('reading ... ',entry)
    hdul2 = fits.open(entry)
    for i in arange(len(hdul1)-1)+1:
      nrows1 = hdul1[i].data.shape[0]
      nrows2 = hdul2[i].data.shape[0]
      nrows = nrows1 + nrows2
      hdu = fits.BinTableHDU.from_columns(hdul1[i].columns, nrows=nrows)
      hdu.header['EXTNAME'] = hdul1[i].header['EXTNAME']
      for colname in hdul1[i].columns.names:
        hdu.data[colname][nrows1:] = hdul2[i].data[colname] 

      if i == 1: 
        hdu1 = hdu 
      else: 
        hdu2 = hdu 

    hdul1 = fits.HDUList([hdu0,hdu1,hdu2])

  hdul1.writeto(output)

  return(None)


#process a single pixel
def do(path,out_path,out_script_path,pixel,expid_range,sdir='',truth=None,nthreads=1,rvpath=None,mwonly=False):
    #get input data files
    datafiles,zbestfiles  = finddatafiles(path,pixel,sdir,rvpath=rvpath)
    if (datafiles == None or zbestfiles == None): return None
	
    #loop over possible multiple data files in the same pixel
    for i in range(len(datafiles)):
        datafile=datafiles[i]
        zbestfile=zbestfiles[i]
	
    #get redshifts
    if zbestfile.find('best') > -1:
        z=readzbest(zbestfile)
    else:
        #Koposov pipeline
        z=readk(zbestfile)
	
    #read primary header and
    #find out if there is FIBERMAP extension
    #identify MWS targets
    hdu=fits.open(datafile)
    enames=extnames(hdu)
    pheader=hdu['PRIMARY'].header
    #print('datafile='+datafile)
    #print('extensions=',enames)
	
    if 'FIBERMAP' in enames: #DESI data
        fibermap=hdu['FIBERMAP']
        suffix=''
        targetid=fibermap.data['TARGETID']
        if 'RA_TARGET' in fibermap.data.names:
            ra=fibermap.data['RA_TARGET']
        else:
            if 'TARGET_RA' in fibermap.data.names:
                ra=fibermap.data['TARGET_RA']
            else:
                ra=-9999.*ones(len(targetid))
        if 'DEC_TARGET' in fibermap.data.names:
            dec=fibermap.data['DEC_TARGET']
        else:
            if 'TARGET_DEC' in fibermap.data.names:
                dec=fibermap.data['TARGET_DEC']
            else:
                dec=-9999.*ones(len(targetid))
        if 'MAG' in fibermap.data.names:
            mag=fibermap.data['MAG']
        else:
            mag=[-9999.*ones(5)]
            for kk in range(len(targetid)-1): mag.append(-9999.*ones(5))
        nspec=ra.size
		
	#set the set of grids to be used
        grids=[]
        for item in range(9): grids.append('n_rdesi'+str(item+1))
        #for item in range(10): grids.append('n_rdesi'+str(item+1))
        #print('grids=',grids)
        maxorder=[3,3,3,2,1,3,3,3,3] #max. order that can be used for interpolations
        #bands=['b']
        bands=['b','r','z']
    else:  #SDSS/BOSS data
        plate=pheader['PLATEID']
        mjd=pheader['MJD']
        suffix="-"+str(mjd)
        #fibermap=hdu['PLUGMAP']
        fibermap=hdu[5]
        fiberid=fibermap.data['fiberid']
        ra=fibermap.data['RA']
        dec=fibermap.data['DEC']
        #mag=zeros((ra.size,5)) # used zeros for LAMOST fibermap.data['MAG']
        mag=fibermap.data['MAG']
        nspec=ra.size
        targetid=[]
        for i in range(nspec):
            targetid.append(str(plate)+'-'+str(mjd)+'-'+str(fiberid[i]))

        targetid=array(targetid)

        #set the set of grids to be used
        #SDSS/BOSS
        grids=[]
        for item in range(9): grids.append('n_rboss'+str(item+1))
	#print(grids)
        maxorder=[3,3,3,2,1,3,3,3,3] #max. order that can be used for interpolations
        bands=['']
        #LAMOST
        #grids=['n_crump3hL']
        #maxorder=[3]
        #bands=['']

    #identify MWS targets:
			
    if mwonly:
        process_target = fibermap.data['MWS_TARGET'] != 0
        #identify targets to process based on redshift: 0.00<|z|<0.01
        #for i in range(nspec):
        #	if z.get(targetid[i],-1) != -1:
        #		if (abs(z[targetid[i]]) < 0.01) & (abs(z[targetid[i]]) > 0.): process_target[i]= True
        process_target = process_target & (fibermap.data["EXPID"] > expid_range[0]) & (fibermap.data['EXPID'] <= expid_range[1])
    else:
        #process_target = ones(nspec, dtype=bool)
        process_target = (fibermap.data["EXPID"] > expid_range[0]) & (fibermap.data['EXPID'] <= expid_range[1])

	
    print('# of processing targets=',np.sum(process_target),'out of ',len(process_target))	
    #skip the rest of the code if there are no targets
    if (process_target.nonzero()[0].size == 0): return None
	
    #set ids array (with targetids) and par dictionary for vrd/ipf file
    ids=[]
    par={}
	
    #truth (optional, for simulations)
    #if (len(sys.argv) == 3):
    npass=0 #count targets that pass the filter (process_target)
    if truth is not None:
        (true_feh,true_teff,true_logg,true_rmag,true_z)=truth
        for k in range(nspec):
            if process_target[k]:
                npass=npass+1
                id=str(targetid[k])
                ids.append(id)
                par[id]=[true_feh[targetid[k]],true_teff[targetid[k]],
                         true_logg[targetid[k]],true_rmag[targetid[k]],
                         true_z[targetid[k]],z[targetid[k]],ra[k],dec[k]]
    else:
        for k in range(nspec):
            if process_target[k]:
                npass=npass+1
                id=str(targetid[k])
                ids.append(id)
                #we patch the redshift here to handle missing redshifts for comm. data from Sergey
                #z[targetid[k]]=0.0
                par[id]=[0.0,0.0,0.0,mag[k][2],0.0,z[targetid[k]],ra[k],dec[k]]

    #collect data for each band
    for j in range(len(bands)):
        if bands[j] == '':
            gridfile=os.environ['GRID_DIR']+"/grids/"+grids[0]+'.dat'
        else:
            gridfile=os.environ['GRID_DIR']+"/grids/"+grids[0]+'-'+bands[j]+'.dat'

        #read grid wavelength array
        x1=lambda_synth(gridfile)
        #read DESI data, select targets, and resample
        (x,y,ivar,r)=readspec(datafile,bands[j])
        ey=sqrt(divide(1.,ivar,where=(ivar != 0.)))
        ey[where(ivar == 0.)]=max(y)*1e3
        nspec, freq = y.shape
        #print('nspec=',nspec)
        #print('n(process_target)=',process_target.nonzero()[0].size)
        y2=zeros((npass,len(x1)))
        ey2=zeros((npass,len(x1)))
        k2=0
        #print('nspec,len(z),npass,len(x1)=',nspec,len(z),npass,len(x1))
        for k in range(nspec):
            if process_target[k]:
                y2[k2,:]=interp(x1,x*(1.-z[targetid[k]]),y[k,:])
                ey2[k2,:]=interp(x1,x*(1-z[targetid[k]]),ey[k,:])
                k2=k2+1

        if (j==0):
            xx=x1
            yy=y2
            eyy=ey2
        else:
            xx=concatenate((xx,x1))
            yy=concatenate((yy,y2),axis=1)
            eyy=concatenate((eyy,ey2),axis=1)

        savetxt(os.path.join(out_script_path,sdir,pixel,pixel)+suffix+'-'+bands[j]+'.wav',x1,fmt='%14.5e')

    savetxt(os.path.join(out_script_path,sdir,pixel,pixel)+suffix+'.wav',xx,fmt='%14.5e')
    fmp = tbl.Table(fibermap.data) [process_target]
    hdu0 = fits.BinTableHDU(fmp)
    hdu0.writeto(os.path.join(out_script_path,sdir,pixel,pixel)+suffix+'.fmp.fits',overwrite=True)
    write_ferre_input(out_script_path,pixel,ids,par,yy,eyy,path=os.path.join(sdir,pixel),suffix=suffix)

    #write slurm scripts
    n_fiber=np.sum(process_target)
    write_slurm(sdir,pixel,out_path,n_fiber,script_path=out_script_path,ngrids=len(grids),nthreads=nthreads, suffix=suffix)

    #loop over all grids
    for k in range(len(grids)):
        #make an array with the names of the synthfiles
        synthfiles=[]
        for j in range(len(bands)):
            if bands[j] == '':
                gridfile=os.environ['GRID_DIR']+"/grids/"+grids[k]+'.dat'
            else:
                gridfile=os.environ['GRID_DIR']+"/grids/"+grids[k]+'-'+bands[j]+'.dat'
            synthfiles.append(gridfile)

        #prepare ferre control file
        nml=mknml(synthfiles,pixel+suffix,k,maxorder[k],nthreads=nthreads)
        writenml(nml,nmlfile='input.nml'+suffix+'_'+str(k),path=os.path.join(out_script_path,sdir,pixel))
        writenml(nml,path=os.path.join(out_script_path,sdir,pixel))
	#print(k)
	#print(nml)
	
    #run ferre
    #ferrerun(path=os.path.join(sdir,pixel))
    #opfmerge(pixel,path=os.path.join(sdir,pixel))
    return None

#find pixels in 'root' directory (spectra-64)
def getpixels(root):
  #root='spectro/redux/dc17a2/spectra-64/'
  d1=os.listdir(root)
  pixels=[]
  sdirs=[]
  for x in d1:
    d2=os.listdir(os.path.join(root,x))
    #d.append(os.path.join(root,x))
    res=[i for i in d2 if '.fits' in i] 
    for y in d2: 
      #d.append(os.path.join(root,x))
      if len(res) == 0: # there are no fits files in the 1st directory, so 2 layer (e.g. DESI)
        #d.append(os.path.join(root,x,y))
        pixels.append(y)
        sdirs.append(x)
      else: 
        entry=os.path.join(root,x)
        if entry not in pixels: pixels.append(entry)  #keep only the first layer (SDSS/BOSS)

  #print(d)
  #print(len(d))
  return sdirs, pixels

#run
def run(pixel,path=None):
  if path is None: path="./"
  #directly run
  #pwd=os.path.abspath(os.curdir)
  #os.chdir(path)
  #job="/bin/bash "+pixel+'.slurm'
  #code=subprocess.call(job)
  #os.chdir(pwd)
  #slurm
  job="sbatch "+os.path.join(path,pixel+'.slurm')
  code=subprocess.call(job)
  #kiko
  #job="kiko "+os.path.join(path,pixel+'.slurm')
  #code=subprocess.call(job)

  return code



if __name__ == "__main__":
	nthreads=4
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--input_files',
		help='Read the list of spectral files from the text file',
		type=str,
		default=None)
	
	parser.add_argument(
		'--input_dir',
		help='directory of spectra64 files',
		type=str,
		default=None)
	
	parser.add_argument(
		'--output_dir',
		help='Output directory for the data tables',
		type=str,
		default=None,
		required=True)
	
	parser.add_argument(
		'--output_script_dir',
		help='Output directory for the slurm scripts and ferre input files',
		type=str,
		default=None,
		required=True)
	parser.add_argument(
		'--allobjects',
		help='Fit all objects, not just MWS_TARGET',
		action='store_true',
		default=False)
	
	parser.add_argument('--minexpid',
		help='Min expid',
		type=int,
		default=None,
		required=False)
	
	parser.add_argument('--maxexpid',
		help='Max expid',
		type=int,
		default=None,
		required=False)
		
	parser.add_argument('--date',
		help='the date which data is processing',
		type=int,
		default=None,
		required=False)		
	
	args = parser.parse_args(sys.argv[1:])
	input_files=args.input_files
	out_path=args.output_dir
	out_script_path = args.output_script_dir
	mwonly=not args.allobjects
	minexpid = args.minexpid
	maxexpid = args.maxexpid
	proc_date = args.date
	
	if minexpid is None:
		minexpid = -1
	if maxexpid is None:
		maxexpid = np.inf
	
	expid_range=[minexpid,maxexpid]
	
	# 'Path' is spectra-64 directory, 'pixels' is a list of pixel number
	if args.input_files:
		pixels = []
		sdirs = []
		with open(input_files, 'r') as fp:
			for l in fp:
				path, f_fits = os.path.split(l.rstrip())
				head, pixel = os.path.split(path)
				path, sdir = os.path.split(head)
				pixels.append(str(pixel))
				sdirs.append(str(sdir))
	if args.input_dir:
		path=args.input_dir
		sdirs,pixels=getpixels(path)
	
	#== Store all the scripts in a folder named with Today's date
	if proc_date is None:
		yr= datetime.date.today().year
		month=datetime.date.today().month
		day=datetime.date.today().day
		now=str(yr*10000+month*100+day)
	else:
		now=proc_date
	
	for pixel,sdir in zip(pixels,sdirs):
		if not os.path.exists(os.path.join(out_path,sdir)): os.mkdir(os.path.join(out_path,sdir))
		if not os.path.exists(os.path.join(out_path,sdir,pixel)): os.mkdir(os.path.join(out_path,sdir,pixel))
		if not os.path.exists(os.path.join(out_script_path,now)): os.mkdir(os.path.join(out_script_path,now))
		if not os.path.exists(os.path.join(out_script_path,now,sdir)): os.mkdir(os.path.join(out_script_path,now,sdir))
		if not os.path.exists(os.path.join(out_script_path,now,sdir,pixel)): os.mkdir(os.path.join(out_script_path,now,sdir,pixel))
	
	#path,out_path,out_script_path,pixel,expid_range,sdir='',truth=None,nthreads=1,rvpath=None,mwonly=False
	
	for pixel,sdir in zip(pixels,sdirs):
		do(path,out_path,os.path.join(out_script_path,now),pixel,expid_range,sdir=sdir,nthreads=nthreads, rvpath=out_path,mwonly=mwonly)
    #run(pixel,path=os.path.join(sdir,pixel))
