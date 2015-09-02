#!/usr/bin/python

import sys, getopt,os
from datetime import date
import pmf,dbread

def showUse():
   print 'test.py -i <inputfile> -o <outputfile>'

def genFolderName():
   today = date.today()
   path = 'run-'+str(today)
   if os.path.isdir(path):
      i=1
      while(os.path.isdir(path+'_'+str(i))):
         i+=1
      path = path+'_'+str(i)
   return path


def main(argv):
   # read the arguments and verify if there are errors
   try:
      opts, args = getopt.getopt(argv,"hi:o:dtep",["ifolder=","ofolder=","datagen","train","evaluate","plot"])
   except getopt.GetoptError as err:
      print err
      showUse()
      sys.exit(2)

   # verify if no arguments are inserted
   if len(opts)==0:
      showUse()
      sys.exit(2)

   ifolder = 'ml-100k'
   ofolder = genFolderName()

   for opt, arg in opts:
      if opt in ("-h", "--help"):
         showUse()
         sys.exit()

      elif opt in ("-i", "--ifile"):
         ifolder = arg

      elif opt in ("-o", "--ofile"):
         ofolder = arg

      elif opt in ("-d", "--datagen"):
         dbread.datagen(ifolder)

      elif opt in ("-t", "--train"):
         if not os.path.isdir(ofolder+'/models'):
            os.makedirs(ofolder+'/models')
         pmf.train(ofolder+'/models')
      '''
      elif opt in ("-e", "--evaluate"):
         if not os.path.isdir(ofolder+'/models'):
            os.makedirs(ofolder+'/models')
         pmf.train(ofolder+'/models')
      '''



if __name__ == "__main__":
   main(sys.argv[1:])
