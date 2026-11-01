"""Create a clean sibling study and execute it without reusing derived results."""
from pathlib import Path
import argparse,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent

def main(output,embedding_python):
    output=output.resolve()
    if output.exists() or output.is_relative_to(ROOT):raise ValueError('Choose a new directory outside this study folder.')
    output.mkdir(parents=True)
    files=['prepare_data.py','core.py','retrieve.py','encode_static.py','analyze.py','test_expanded.py','validate_results.py','download_inputs.py','protocol_v2.json','protocol_v3.json','static_model_manifest.json','data_provenance.json']
    for f in files:shutil.copy2(ROOT/f,output/f)
    data=output/'scratch/data';models=output/'scratch/models';cache=output/'scratch/embeddings'
    def run(py,script,*args):subprocess.run([py,str(output/script),*map(str,args)],check=True,cwd=output)
    run(sys.executable,'download_inputs.py','--data',data,'--models',models)
    run(sys.executable,'prepare_data.py','--data',data)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(output),'-p','test_expanded.py','-v'],check=True)
    for kind in ['potion_base','potion_retrieval']:run(embedding_python,'encode_static.py','--data',data,'--models',models,'--cache',cache,'--kind',kind)
    run(sys.executable,'retrieve.py','--data',data,'--cache',cache)
    run(sys.executable,'analyze.py','--data',data)
    run(sys.executable,'validate_results.py','--data',data)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--embedding-python',default=sys.executable);a=p.parse_args();main(a.output,a.embedding_python)
