def read_sequence(fastafile):
    '''Returns the whole sequence of a fasta file in only one line'''
    with open(fastafile,'r') as f:
        f.readline()  
        seq = ''
        for line in f.readlines():
            seq += line.strip('\n').upper()
    return seq

chr_genome1 = read_sequence('Osat_IR64_AGI_NSD_chrOK.id_chr01.fasta')
chr_genome2 = read_sequence('Osat_Azucena_AGI_chrOK.id_chr01.fasta')

input_file = open('NAM_Azucena_chr_all_imputed_chr01.txt','r')
output_file = open('imputed_recombination_chr01.txt','w')

w_size = 100_000
w_end = w_size

lines = input_file.readlines()
i=0
while i < len(lines)-1:
    
    if lines[i][0]=='*':
        pos1, samples1 = lines[i].split()
        pos2, samples2 = lines[i+1].split()
        name1,pos1 = pos1.split('_')
        name2,pos2 = pos2.split('_')
        pos1, pos2 = int(pos1), int(pos2)
        
        while w_end < pos1:
            output_file.write(name1+'?_'+str(w_end)+' '+samples1+'\n')
            w_end += w_size
        
        output_file.write(lines[i])
        if pos1 == w_end:
            w_end += w_size
        
        while w_end < pos2:
            if w_end-pos1 < pos2-w_end:
                output_file.write(name1+'?_'+str(w_end)+' '+samples1+'\n')
            else:
                output_file.write(name2+'?_'+str(w_end)+' '+samples2+'\n')
            w_end += w_size

    else:
        output_file.write(lines[i])
    
    i+=1

output_file.write(lines[i])

while w_end < max(len(chr_genome1),len(chr_genome2)):
    output_file.write(name2+'?_'+str(w_end)+' '+samples2+'\n')
    w_end += w_size

input_file.close()
output_file.close()
