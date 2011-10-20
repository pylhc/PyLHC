
import os,sys



def get_from_timber(data, filename,fillnumber=0,arguments='',overwrite=False,dataformat='TSV',minfilesize=1000):
    '''
     Get a set of data from TIMBER and store it in file filename.dataformat
     
     :param list data: Can also be comma-separated string. List of variable names to extract.
     :param string filename: Name of file where data is stored, as filename.dataformat
     :param integer fillnumber: Which fill number to extract from, otherwise latest data.
     :param string arguments: Additional arguments to pass to the java command (expert option)
     :param bool overwrite: if True, overwrites file if it already exist.
     :param string dataformat: Type of formatting in the output file.
     :param int minfilesize: Minimum size of file in bytes for successful extraction (expert option)
    '''
    dataformat=dataformat.upper()
    if dataformat not in ['TSV','CSV']:
        raise ValueError("%s is not a valid dataformat" % dataformat)
    
    if not overwrite:
        if os.path.isfile(get_file_directory()+filename+'.'+dataformat):
            if os.path.getsize(os.path.join(get_file_directory(),filename+'.'+dataformat))>minfilesize:
                if sys.flags.debug:
                    print "INFO: Not creating file %s.%s, it already exist" % (filename,dataformat)
                return None
    # you need to set this yourself.. 
    # Relative to extraction folder OR absolute paths
    CLASSPATH=_get_classpath(['../../lib/java/',
                              '../../../javalibs/',
                              '../../lib/',
                              '/afs/cern.ch/user/y/ylevinse/javalibs/'])
    
    cmd='java -Xms128m -Xmx2048m -cp %s cern.accdm.timeseries.access.client.commandline.Export' % CLASSPATH
    if type(data)==list:
        data=','.join(data)
    if fillnumber:
        cmd+=' -M DS -fn %i' % fillnumber
    else:
        cmd+=' -M LD'
    cmd+=' -N %(filename)s -F %(dataformat)s -vs %(data)s %(arguments)s' % locals()
    _runCmd(cmd)
    if os.path.getsize(os.path.join(get_file_directory(),filename+'.'+dataformat))<minfilesize:
        raise ValueError("%s is too small" % filename)

def get_file_directory():
    '''
     Returns folder path where output files are stored
    '''
    path=os.path.dirname(__file__)+'/'
    for l in file(path+'configuration.properties'):
        if l.split('=')[0]=='FILE_DIRECTORY':
            return path+l.split('=')[1].strip()

def cat_files(filelist,filename):
    '''
     Concatenate files in filelist into new file filename
    '''
    cmd="cat "
    fld=get_file_directory()
    print "DBG",filelist
    for f in filelist:
        cmd+=fld+f+" "
    cmd+=" > "+fld+filename
    _runCmd(cmd)


def _runCmd(cmd):
    '''
     Runs specified command in the folder where this
     particular module is located.
     
     Checks that the command returns 0.
    '''
    path=os.path.dirname(__file__)+'/'
    if sys.flags.debug:
        print( "Running shell command: "+cmd)
    ret=os.system("cd "+path+";"+cmd)
    if ret!=0:
        raise ValueError("Shell command returned %i" % ret)
        
def _get_classpath(directory):
    path=os.path.dirname(__file__)+'/'
    if type(directory)==list:
        for d in directory:
            if d.strip()[0]!='/':
                d=path+d
            if os.path.isdir(d):
                if d[-1]!='/':
                    d+='/'
                directory=d
                break
    ret=''
    for jar in os.listdir(directory):
        if jar[-4:]=='.jar':
            ret+=directory+jar+':'
    if len(ret)>0:
        return ret[:-1]
    return ret
    
