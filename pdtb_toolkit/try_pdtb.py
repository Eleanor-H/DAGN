'''

Date: 25/10/2020
Author: Yinya Huang
'''


from pdtb_toolkit.pdtb import CorpusReader, Datum



if __name__ == '__main__':

    output_dir = './output_pdtb_explicit_rels.txt'

    pdtb_dir = '/Users/mac/Documents/__datasets/PDTB2/pdtb2.csv'

    pdtb = CorpusReader(pdtb_dir)

    with open(output_dir, 'a') as f:
        i = 0
        for datum in pdtb.iter_data():
            # print(datum.Relation)
            # assert 1 == 0
            if datum.Relation == 'Explicit':
                f.write('\n' + '='*20)
                # for k,v in datum.__dict__.items():
                #     print('{}: {}'.format(k,v))
                f.write('Connective_RawText: {}\n'.format(datum.Connective_RawText))
                f.write('ConnHead: {}\n'.format(datum.ConnHead))
                f.write('Conn1: {}\n'.format(datum.Conn1))
                f.write('Conn2: {}\n'.format(datum.Conn2))
                f.write('ConnHeadSemClass1: {}\n'.format(datum.ConnHeadSemClass1))  # semantic class
                f.write('ConnHeadSemClass2: {}\n'.format(datum.ConnHeadSemClass2))  # semantic class
                f.write('Conn2SemClass1: {}\n'.format(datum.Conn2SemClass1))
                f.write('Conn2SemClass2: {}\n'.format(datum.Conn2SemClass2))

                f.write('Attribution_RawText: {}\n'.format(datum.Attribution_RawText))
                f.write('Arg1_RawText: {}\n'.format(datum.Arg1_RawText))
                f.write('Arg2_RawText: {}\n'.format(datum.Arg2_RawText))
                f.write('FullRawText: {}\n'.format(datum.FullRawText))
                f.write('\n')
                i += 1
            # if i == :
            #     break



'''
save datum.Connective_RawText,

group them by datum.ConnHead.

The edge direction taken from datum.ConnHeadSemClass1 and datum.ConnHeadSemClass2 

'''