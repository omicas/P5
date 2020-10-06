import pandas as pd
from tqdm import tqdm

def read_sequence(fastafile):
    '''Returns the whole sequence of a fasta file as a str'''
    with open(fastafile,'r') as f:
        f.readline()  
        seq = ''
        for line in f.readlines():
            seq += line.strip('\n').upper()
    return seq

# read reference genome
rbases = read_sequence('Osat_IR64_AGI_NSD_chrOK.id_chr01.fasta')
# read query genome
qbases = read_sequence('Osat_Azucena_AGI_chrOK.id_chr01.fasta')

# read coordinates file
coords = pd.read_csv('coords_IR64_Azucena_chr01.txt',sep='\t')
coords.rename(columns={'S1': 'Sr', 'E1': 'Er', 'S2': 'Sq', 'E2': 'Eq', 'LEN1':'len_r', 'LEN2':'len_q'}, inplace=True)

# detect inverted contigs in query genome (there are no inversions in reference genome)
coords['inversion_q'] = coords.apply(lambda df: df.Sq >= df.Eq, axis=1)

# number of inverted contigs in query genome
# inv_c = coords[coords['inversion_q']==True].inversion_q.count()
# print('There are {0} inverted contigs in query genome'.format(inv_c))

# sort dataframe by reference contigs
coords = coords.sort_values(by=['Sr','Er'])
coords = coords.reset_index()

# read variants file
variants_df = pd.read_csv('mum_variants_IR64_Azucena_chr01.vcf',sep='\t')

# identify snps and variants (indels and delis) per position in reference genome
print('mapping snps and variants in reference genome...')
snp = [0 for _ in range(len(rbases))]
var = [0 for _ in range(len(rbases))]

for i in tqdm(range(variants_df.shape[0])):
    x = len(variants_df['REF'][i])     	# number of nucleotides in reference genome
    y = len(variants_df['ALT'][i])		# number of nucleotides in query genome
    pos = variants_df['POS'][i]-1		# position of variant in reference genome
    if x-y == 0:
        snp[pos] = 1					# snp: x = y
    else:
        var[pos] = x-y 					# delin: x < y , indel: x > y

print('mapping query genome in reference genome...')
# mapping process
qpos  = [0 for _ in range(len(rbases))]		#qpos \in [1,len(qbases)]
qbase = ['-' for _ in range(len(rbases))]	#qbases \in {'A','G','T','C'}
inv   = [-1 for _ in range(len(rbases))]	#inv \in {True, False}


# for each row (contig) in coordinates file
for i in tqdm(range(coords.shape[0])):
    Sr = coords['Sr'][i]-1
    Er = coords['Er'][i]-1
    
    inv[Sr:Er+1]= [coords['inversion_q'][i]] * (Er+1 - Sr)  # whole contig have the same inversion value
    
    if inv[Sr] == 0: delta = 1
    else: delta = -1
    
    if qpos[Sr] == 0: 								# if the Sr position has not been mapped
        Sq = coords['Sq'][i]  							# Sr maps to Sq
    
    else: 											# if Sr position is already mapped
        if coords['%IDY'][i-1] > coords['%IDY'][i]:		# if mapped contig has a higher %IDY
            Sr_new = coords['Er'][i-1] + 1  				# start mapping from Er[i-1]+1
            Sq = Sr_new - Sr + coords['Sq'][i]           	# Er[i-1]+1 maps to (Er[i-1]+1 - Sr) + Sq
            Sr = Sr_new                     				# continue mapping from there (here is possible that Sr > Er)
        else:											# if mapped contig has lower %IDY
            Sq = coords['Sq'][i]      						# Sr maps to Sq
    
    if Sr <= Er:									# map first position only if Sr <= Er 
    	qpos[Sr] = Sq 									# Sr maps to Sq
    	qbase[Sr] = qbases[qpos[Sr]-1]					# find corresponding query nucleotide
    
    j = Sr
    while j < Er:									# map the rest of the contig
        m = max(1,var[j]+1)					
        qpos[j+m] = qpos[j] + delta*(1 - var[j+m-1])               
        qbase[j+m] = qbases[qpos[j+m]-1]
        j += m

print('creating dataframe...')
# create dataframe
df_ref = pd.DataFrame({'rbase': list(rbases)})
df_ref.insert(0,'rpos', df_ref.index + 1)
df_ref['qpos'] = qpos
df_ref['qbase'] = qbase
df_ref['inversion'] = inv
df_ref['snp'] = snp
df_ref['variant'] = var
df_ref['identical'] = [1 if q==r else 0 for q,r in zip(qbase,rbases)]

print('saving dataframe...')
# save dataframe as csv
df_ref.to_csv('map_ref_chr01.csv',sep=',',compression="gzip")

print('Done...')