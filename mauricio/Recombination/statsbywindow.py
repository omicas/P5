import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



with open('ref_qry_names.txt') as file:      # Calls the names of reference and query fasta files present in the ref_qry_names file
    names = file.readlines()                 # Read lines in this file
    
rname = names[0]                             # Object with the reference fasta name
rname = rname[: - 1]                         # remove last  characters of first line "/n"
qname = names[1]                             # Object with the query fasta name


#-----------------------------------------------------------------------------------------------------------------------
#Read sequences in fasta files and transform them in dataframes by base
#-----------------------------------------------------------------------------------------------------------------------
# Code to read fasta file with header
def read_fasta(fp):
       name, seq = None, []
       for line in fp:
           line = line.rstrip()
           if line.startswith(">"):
               if name: yield (name, ''.join(seq))
               name, seq = line, []
           else:
               seq.append(line)
       if name: yield (name, ''.join(seq))
        
#--------------------------------------------------------------------------------------------------------------------------
# Code to import reference and query files a transform them in data frames

with open('{0}'.format(rname)) as fp:       #Import a reference file
       for name, seq in read_fasta(fp):     #Read the fasta´s file name and sequence       
           rbases = list(seq)               #Generates a "rbase" list with the data from the reference sequence 
rl = len(rbases)                            #Extracts the sequence´s length
rp= list(range(1,rl+1))                     #Generates a "rp" list with numbers from 1 to the total length of the reference sequence
rp = pd.DataFrame(rp)                       #Generates a data frame from "rp" list
rbases = pd.DataFrame(rbases)               #Generates a data frame from "rbases" list
reftable = pd.concat([rp, rbases], axis=1)  #Generates a table joinig the "rp" and "rbase" dataframes
reftable.columns = ['rpos','rbase']         #Assign names to table columns 

with open('{0}'.format(qname)) as fp:       #Import a query file
       for name, seq in read_fasta(fp):     #Read the fasta´s file name and sequence       
           qbases = list(seq)               #Generates a "baseq" list with the data from the query sequence 
ql = len(qbases)                            #Extracts the sequence´s length
qp= list(range(1,ql+1))                     #Generates a "pq" list with numbers from 1 to the total length of the query sequence
qp = pd.DataFrame(qp)                       #Generates a data frame from "pq" list
qbases = pd.DataFrame(qbases)               #Generates a data frame from "basesq" list
qrytable = pd.concat([qp, qbases], axis=1)  #Generates a table joinig the pq" and "baseq" dataframes
qrytable.columns = ['qpos','qbase']         #Assign names to table columns 

#------------------------------------------------------------------------------------------------------------------------
#Extracts GC content for reference and query dataframes
#------------------------------------------------------------------------------------------------------------------------

rg = reftable[(reftable.rbase == "g") ]        # Extracts base g rows from reference sequence
rG = reftable[(reftable.rbase == "G") ]        # Extracts base G rows from reference sequence
rc = reftable[(reftable.rbase == "c") ]        # Extracts base c rows from reference sequence
rC = reftable[(reftable.rbase == "C") ]        # Extracts base C rows from reference sequence
rgG = pd.concat([rg, rG])                      # Concatenates bases rg and rG rows in a data frame
rcC = pd.concat([rc, rC])                      # Concatenates bases rc and rC rows in a data frame
rGC = pd.concat([rgG, rcC])                    # Concatenates  rgG and rcC data frames in a new data frame rGC 

qg = qrytable[(qrytable.qbase == "g") ]        # Extracts base g rows from query sequence
qG = qrytable[(qrytable.qbase == "G") ]        # Extracts base G rows from query sequence
qc = qrytable[(qrytable.qbase == "c") ]        # Extracts base c rows from query sequence
qC = qrytable[(qrytable.qbase == "C") ]        # Extracts base C rows from query sequence
qgG = pd.concat([qg, qG])                      # Concatenates bases qg and qG rows in a data frame
qcC = pd.concat([qc, qC])                      # Concatenates bases qc and qC rows in a data frame
qGC = pd.concat([qgG, qcC])                    # Concatenates  qgG and qcC data frames in a new data frame qGC

snps = pd.read_csv("mum_filterX.snps.vcf", sep="\t")          #Charge the snps file as data frame
snps.columns = ["pos","snp"]                      #Rename columns in snps data frame

indels = pd.read_csv("pindels.vcf", sep="\t")      #Charge the indels file as data frame
indels.columns = ["rpos","indels"]                 #Rename columns in indels data frame

#-------------------------------------------------------------------------------------------------------------------------
# Count of parameters by window
#-------------------------------------------------------------------------------------------------------------------------
# Set the window which you want to work. Corresponds to the interval of bases, example: 1000, 2500, 10000, 250000, 1000000, etc.

window = 1000                                      #Set the window to do statistics

t = rl+window                                      #Object with data length plus one window to do bins
w = list(range(0, t, window))                      #A list of bins based on the selected window

#-------------------------------------------------------------------------------------------------------------------------

rGChistdata = np.histogram(rGC['rpos'], bins = w)          #Extracts the GC count from the reference data based on the window
rGCint=rGChistdata[1]                                      #Object with intervals
rGCcount=rGChistdata[0]                                    #Object with GC counts
rGCint = pd.DataFrame(rGCint,columns=['window'])           #Converts rGCint in a data frame
rGCcount = pd.DataFrame(rGCcount,columns=['rGC count'])    #Converts rGCcount in a data frame
rGCtable = pd.concat([rGCint, rGCcount], axis=1)           #Generates a data frame that merges rGCint and rGCcount

qGChistdata = np.histogram(qGC['qpos'], bins = w)          #Extracts the GC count from the query data based on the window
qGCint=qGChistdata[1]                                      #Object with intervals
qGCcount=qGChistdata[0]                                    #Object with GC counts
qGCint = pd.DataFrame(qGCint,columns=['window'])           #Converts qGCint in a data frame
qGCcount = pd.DataFrame(qGCcount,columns=['qGC count'])    #Converts qGCcount in a data frame
qGCtable = pd.concat([qGCint, qGCcount], axis=1)           #Generates a data frame that merges qGCint and qGCcount

snpshistdata = np.histogram(snps['rpos'], bins = w)
snpsint = snpshistdata[1]
snpscount = snpshistdata[0]
snpsint = pd.DataFrame(snpsint,columns=['window'])
snpscount = pd.DataFrame(snpscount,columns=['snps count'])
snpstable = pd.concat([snpsint, snpscount], axis=1)

indelshistdata = np.histogram(indels['rpos'], bins = w)
indelsint = indelshistdata[1]
indelscount = indelshistdata[0]
indelsint = pd.DataFrame(indelsint,columns=['window'])
indelscount = pd.DataFrame(indelscount,columns=['indels count'])
indelstable = pd.concat([indelsint, indelscount], axis=1)




rtotaltable = pd.concat([rGCint, snpscount, indelscount,  rGCcount, qGCcount], axis=1) 
rtotaltable.to_csv(r'rtotaltable.csv', index = False)



plt.hist(snps['rpos'],  bins=w, histtype='step', color= 'pink')
plt.ylabel('Count')
plt.xlabel('Chromosome length')
plt.title('SNPs')
plt.savefig("SNPs.jpg")

plt.hist(indels['rpos'],  bins=w, histtype='step', color= 'orange')
plt.ylabel('Count')
plt.xlabel('Chromosome length')
plt.title('Indels')
plt.savefig("Indels.jpg")



plt.hist(rGC['rpos'],  bins=w, histtype='step', color= 'red')
plt.ylabel('Count')
plt.xlabel('Chromosome length')
plt.title('rGC count')
plt.savefig("rGC count.jpg")

plt.hist(qGC['qpos'],  bins=w, histtype='step', color= 'blue')
plt.ylabel('Count')
plt.xlabel('Chromosome length')
plt.title('qGC count')
plt.savefig("qGC count.jpg")

plt.hist(rGC['rpos'],  bins=w, histtype='step', color= 'red')
plt.hist(qGC['qpos'],  bins=w, histtype='step', color= 'blue')

plt.hist(snps['rpos'],  bins=w, histtype='step', color= 'pink')
plt.hist(indels['rpos'],  bins=w, histtype='step', color= 'orange')
plt.ylabel('Count')
plt.xlabel('Chromosome length')
plt.title('All Parameters')
plt.savefig("All parameters.jpg")

plt.hist(snps['rpos'],  bins=w, histtype='step', color= 'pink')

plt.hist(indels['rpos'],  bins=w, histtype='step', color= 'orange')
plt.ylabel('Count')
plt.xlabel('Chromosome length')
plt.title('All Parameters less GC count')
plt.savefig("Snps and Indels.jpg")