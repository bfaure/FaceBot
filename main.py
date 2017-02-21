import datetime
import sys
import tensorflow as tf 
from time import time
import random
import numpy as np

from keras.models import Sequential
from seq2seq.models import AttentionSeq2Seq, SimpleSeq2Seq


# numbers of unique words in all queries and responses
QUERY_MAPPING_SIZE = 0
RESPONSE_MAPPING_SIZE = 0
SEQUENCE_LENGTH = 100
BATCH_SIZE = 32
LOAD_PRIOR_MODEL = False

#======================================
# Utilities for parsing messages.htm to message_t list

def get_int_from_month(month):
	months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
	return months.index(month)+1

class message_t:

	def __init__(self,message_str):
		self.message_str = message_str
		self.parse_user()
		self.parse_meta()
		self.parse_text()
		self.check_for_url()

		self.fix_text()
		self.fix_user()
		self.fix_timestamp()

		self.process_data()

	def fix_timestamp(self):
		# gets seconds since epoch from the timestamp string
		self.timestamp = self.timestamp.replace('\n'," ") # remove any trailing endlines

		broken = self.timestamp.split(",")
		month = get_int_from_month(broken[1].split(" ")[1])
		day = int(broken[1].split(" ")[2])
		year = int(broken[2].split(" ")[1])
		hour = int(broken[2].split(" ")[3].split(":")[0])
		minute = broken[2].split(" ")[3].split(":")[1]
		if minute.find("pm")!=-1: # PM
			hour+=12
			if hour==24: hour=23
		else: # AM
			if hour==12: hour = 0

		minute = int(minute[:2])
		date = datetime.datetime(year,month,day,hour,minute)
		self.time_in_seconds = (date-datetime.datetime(1970,1,1)).total_seconds()

	def check_for_url(self):
		# checks to see if the text is just a URL
		if self.text.find("http://")!=-1:
			self.is_url = True
		elif self.text.find(".com")!=-1:
			self.is_url = True
		else:
			self.is_url = False

	def fix_user(self):
		# if the username contains an @
		self.user = self.user.replace("&#064;","@")

		# check if the name is a facebook email and swap it out with the real name if so
		emails 	= ["1221627603@facebook.com"]
		users 	= ["Brian Faure"]

		for email,user in list(zip(emails,users)):
			if self.user == email:
				self.user = user
				break	 

	def fix_text(self):
		# swaps out any known Facebook character codes with their appropriate characters
		char_codes 	= ["&#039;","&#123;","&#125;", "\n"] 
		chars 		= ["\'","{","}"," "]
		for code,correct in list(zip(char_codes,chars)):
			self.text = self.text.replace(code,correct)

		if self.is_url==False: # more stuff to remove if not containing url
			char_codes 	= ["&quot;","&lt;",	"&gt;","&#064"] 
			chars 		= ["\"","<",	">","@"]
			for code,correct in list(zip(char_codes,chars)):
				self.text = self.text.replace(code,correct)

			self.text = self.text.replace("."," ") # replace periods with spaces
			self.text = self.text.replace(","," ") # replace commas with spaces
			self.text = self.text.replace("?"," ") # replace question marks with spaces
			self.text = self.text.replace("-"," ")
			self.text = self.text.replace("/"," ")
			self.text = self.text.replace("^"," ")
			self.text = self.text.replace("["," ")
			self.text = self.text.replace("*"," ")
			self.text = self.text.replace("+"," ")
			self.text = self.text.replace("]"," ")
			self.text = self.text.replace("["," ")
			self.text = self.text.replace("("," ")
			self.text = self.text.replace(")"," ")
			self.text = self.text.replace("="," ")
			self.text = self.text.replace("$"," ")
			self.text = self.text.replace(":"," ")
			self.text = self.text.replace("!"," ")
			self.text = self.text.replace("_"," ")
			self.text = self.text.replace("<"," ")
			self.text = self.text.replace(">"," ")
			self.text = self.text.replace("{","")
			self.text = self.text.replace("}","")
			self.text = self.text.replace("'","")
			self.text = self.text.replace("\"","")
			self.text = self.text.replace("   "," ") # trim useless spaces
			self.text = self.text.replace("  "," ") # trim useless spaces

	def process_data(self):
		# whatever we want to do after parsing when initialized...

		# calculate the longest item and its length
		longest = len(self.user)
		if len(self.timestamp)>longest: longest = len(self.timestamp)
		if len(self.text)>longest: longest = len(self.text)
		self.length = longest

		# calculate things used for printing
		horizontal_border_length = self.length+12
		self.horizontal_border = ""
		for _ in range(horizontal_border_length+2):
			self.horizontal_border += "="

		self.user_print_string 		= "|  User:   "+self.user 
		if len(self.user_print_string)<horizontal_border_length:
			for i in range(horizontal_border_length-len(self.user_print_string)):
				self.user_print_string += " "
		self.user_print_string += " |"

		self.timestamp_print_string = "|  Meta:   "+self.timestamp
		if len(self.timestamp_print_string)<horizontal_border_length:
			for i in range(horizontal_border_length-len(self.timestamp_print_string)):
				self.timestamp_print_string += " "
		self.timestamp_print_string += " |"

		self.text_print_string 		= "|  Text:   "+self.text 
		if len(self.text_print_string)<horizontal_border_length:
			for i in range(horizontal_border_length-len(self.text_print_string)):
				self.text_print_string += " "
		self.text_print_string += " |"

	def parse_user(self):
		# parses out the sender of the message
		self.user = self.message_str[:self.message_str.find("</span>")]

	def parse_meta(self):
		# parses out the timestamp data from the message
		timestamp_start = "<span class=\"meta\">"
		timestamp_end = "</span>"
		start_index = self.message_str.find(timestamp_start)+len(timestamp_start)
		end_index = self.message_str.find(timestamp_end,start_index)
		self.timestamp = self.message_str[start_index:end_index]

	def parse_text(self):
		# parses out the actual message text
		text_start = "</span></div></div><p>"
		text_end = "</p>"
		start_index = self.message_str.find(text_start)+len(text_start)
		end_index = self.message_str.find(text_end,start_index)
		self.text = self.message_str[start_index:end_index]

	def display(self):
		# prints the message to terminal
		print(self.horizontal_border)
		print(self.user_print_string)
		print(self.timestamp_print_string)
		print(self.text_print_string)
		#print("|  Time: "+str(self.time_in_seconds)+" |")
		print(self.horizontal_border)

def read_messages(filename,count=-1,display=False):
	f = open(filename,"r", encoding='utf8')
	text = f.read()
	print("\nReading messages...")
	messages = text.split("<div class=\"message\"><div class=\"message_header\"><span class=\"user\">")
	print("Found "+str(len(messages))+" messages in "+filename)
	parsed_messages = []
	if count==-1: count=1000000
	print("\nParsing messages...",end="\r")
	i=0
	messages = messages[1:] # first entry isnt a message

	for message in messages:
		if i>count: break
		print("Parsing messages... "+str(i),end="\r")
		m = message_t(message)
		if m.text not in [""," ","  "]:
			parsed_messages.append(m)
			i+=1

	print("\nFinished parsing "+str(len(parsed_messages))+" messages.")

	if display:
		num_to_print = 3
		print("Printing the first "+str(num_to_print)+" messages...")
		for j in range(num_to_print):
			parsed_messages[j].display()
	return parsed_messages

def save_messages(filename,message_ts):
	print("Saving "+str(len(message_ts))+" messages to "+filename+"...")
	f = open(filename,"w",encoding='utf8')
	for message in message_ts:
		f.write(message.horizontal_border+"\n")
		f.write(message.user_print_string+"\n")
		f.write(message.timestamp_print_string+"\n")
		f.write(message.text_print_string+"\n")
		f.write(message.horizontal_border+"\n\n")
	print("Finished saving "+str(len(message_ts))+" messages.")

def sort_messages(message_ts):
	# sorts the messages by their time_in_seconds value
	print("\nSorting messages...")
	sorted_messages = sorted(message_ts, key=lambda message: message.time_in_seconds)
	print ("Finished sorting.")
	return sorted_messages

#======================================
# Utilities to parse a sorted message_t list into query-response tuples where
# the query is the message sent to the person the Bot is emulating and the response
# is the next message sent from the person emulating the Bot.

def create_query_response_pairs(messages,main_person,is_sorted=False,display=False):
	print("\nCreating query-response pairs...")
	# check if the list is sorted, if not, sort it
	if is_sorted==False: messages = sort_messages(messages)

	# list of tuples, each tuple contains two strings, the first being the string query
	# sent to the main_person and the second being the main person's response
	pairs = []

	current_query = "None"
	current_query_time = -1

	print("Reading messages...",end="\r")
	for message in messages:
		print("Reading messages... "+str(messages.index(message)),end="\r")
		if message.user==main_person and current_query!="None":
			new_pair = [current_query,message.text]
			pairs.append(new_pair)
			current_query = "None"
		else:
			current_query = message.text 
			current_query_time = message.time_in_seconds

	print("\nDone creating "+str(len(pairs))+" query-response pairs.")

	if display==True:
		num_to_print = 3
		if len(pairs)<num_to_print: num_to_print = len(pairs)
		print("Printing the first "+str(num_to_print)+" query-response pairs...")
		print("===========================================================")
		for i in range(num_to_print):
			pair = pairs[i]
			print("To "+main_person+":\t "+pair[0])
			print("Response:\t "+pair[1])
		print("===========================================================")
	return pairs

def get_word_id(word,mappings):
	# takes in a word and checks to see if it is already in the mapping, if so
	# it will return the integer id of the word, if not return -1
	word = word.lower()

	if word.find("http")!=-1: return -2
	if word.find(".com")!=-1: return -2

	if word in [""," "]: return -2 # indicate that this is not a word, drop it
	for mapped_word,word_id in mappings:
		if mapped_word==word: return word_id
	return -1

def tokenize_query_response_pairs(pairs,display=False):
	global QUERY_MAPPING_SIZE
	global RESPONSE_MAPPING_SIZE

	print("\nTokenizing "+str(len(pairs))+" query-response pairs...")
	# takes in query-response pairs and tokenizes the words so that each 
	# word is replaced by a unique integer id

	# list of tuples, first element of tuple is going to be a word
	# and the second element is the associated integer id mapping
	mappings = []
	tokenized_pairs = [] # tokenized version of input
	current_id = 0 # first mapping will be some word to 0

	print("Tokenizing pair...",end="\r")
	# iterate over each pair in the input and tokenize it
	for pair in pairs:
		print("Tokenizing pair... "+str(pairs.index(pair)),end="\r")
		tokenized_pair = []
		tokenized_query = []
		tokenized_response = []

		query = pair[0]
		response = pair[1]

		query_words = query.split() # split into words
		response_words = response.split() # split into words

		for word in query_words:
			word_id = get_word_id(word,mappings)
			if word_id==-2: continue
			elif word_id==-1: # create new mapping
				QUERY_MAPPING_SIZE+=1
				tokenized_query.append(current_id)
				new_mapping = [word.lower(),current_id]
				mappings.append(new_mapping)
				current_id+=1
			else: tokenized_query.append(word_id)

		for word in response_words:
			word_id = get_word_id(word,mappings)
			if word_id==-2: continue
			elif word_id==-1: # create new mapping
				RESPONSE_MAPPING_SIZE+=1
				tokenized_response.append(current_id)
				new_mapping = [word.lower(),current_id]
				mappings.append(new_mapping)
				current_id+=1
			else: tokenized_response.append(word_id)

		tokenized_pair = [tokenized_query,tokenized_response]
		tokenized_pairs.append(tokenized_pair)

	print("\nDone tokenizing "+str(len(tokenized_pairs))+" query-reponse pairs.")
	print("Created "+str(len(mappings))+" word-integerid mappings.")

	if display:
		num_to_print = 3
		if len(tokenized_pairs)<num_to_print: num_to_print = len(tokenized_pairs)
		print("Printing the first "+str(num_to_print)+" tokenized query-response pairs...")
		print("===========================================================")
		for i in range(num_to_print):
			pair = tokenized_pairs[i]
			print("Query:     ",pair[0])
			print("Response:  ",pair[1])
		print("===========================================================")
	return tokenized_pairs,mappings

def save_tokenized_pairs_and_mappings(tokenized_pairs,mappings):
	print("\nSaving "+str(len(tokenized_pairs))+" tokenized pairs and "+str(len(mappings))+" mappings...")

	f1 = open("data/tokenized_pairs.txt","w",encoding='utf8')
	f2 = open("data/mappings.txt","w",encoding='utf8')

	print("Saving tokenized pairs to data/tokenized_pairs.txt...")
	for query,response in tokenized_pairs:
		f1.write("{INPUT:[")
		for item in query:
			f1.write(str(item))
			if query.index(item)!=len(query)-1:
				f1.write(",")
		f1.write("]}\n")

		f1.write("{OUTPUT:[")
		for item in response:
			f1.write(str(item))
			if response.index(item)!=len(response)-1:
				f1.write(",")
		f1.write("]}\n")

	print("Saving mappings to data/mappings.txt...")
	for string,integer_id in mappings:
		f2.write("{STRING:\""+string+"\",ID:"+str(integer_id)+"}\n")
	print("Finished saving tokenized pairs and mappings.")

def get_data(filename,verbose=False,load_prior=False):
	if load_prior:
		print("\nLoading data...")
		mappings = open("data/mappings.txt","r",encoding='utf8').read().split("\n")
		tokenized_pairs = open("data/tokenized_pairs.txt","r",encoding='utf8').read().split("\n")

		parsed_mappings = []
		parsed_pairs = []

		print("Loading mappings...",end="\r")
		for mapping in mappings:
			print("Loading mappings... "+str(mappings.index(mapping)),end="\r")
			if mapping in [""," "]: continue
			word = mapping[9:mapping.find("\"",9)]
			word_id = int(mapping[mapping.find("ID:")+3:mapping.find("}")])
			parsed_mappings.append([word,word_id])
		print("\nLoading tokenized pairs...",end="\r")
		temp_pair = []
		for pair in tokenized_pairs:
			print("Loading tokenized pairs..."+str(tokenized_pairs.index(pair)),end="\r")
			if pair in [""," "]: continue
			if pair.find("INPUT")!=-1:
				query = pair[8:pair.find("]}")].split(",")
				query_nums = []
				for i in range(len(query)):
					if query[i]!="": query_nums.append(int(query[i]))
				temp_pair.append(query_nums)
			elif pair.find("OUTPUT")!=-1 and len(temp_pair)==1:
				response = pair[9:pair.find("]}")].split(",")
				response_nums = []
				for i in range(len(response)):
					if response[i]!="": response_nums.append(int(response[i]))
				temp_pair.append(response_nums)
				parsed_pairs.append(temp_pair)
				temp_pair = []

		print("\nFinished loading "+str(len(parsed_mappings))+" mappings and "+str(len(parsed_pairs))+" query-response pairs.")
		return parsed_pairs,parsed_mappings

	dest = "data/messages.txt"
	messages = read_messages(filename,count=-1,display=verbose)
	messages = sort_messages(messages)
	save_messages(dest,messages)
	pairs = create_query_response_pairs(messages,"Brian Faure",is_sorted=True,display=verbose)
	tokenized_pairs,mappings = tokenize_query_response_pairs(pairs,display=verbose)
	save_tokenized_pairs_and_mappings(tokenized_pairs,mappings)
	return tokenized_pairs,mappings 

def split_to_encoded_decoded(tokenized_pairs):
	# splits the tokenized pairs into inputs and outputs
	encoded = []
	decoded = []
	for enc,dec in tokenized_pairs:
		encoded.append(enc)
		decoded.append(dec)
	return encoded,decoded 


_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

def put_pairs_in_buckets(tokenized_pairs):
	# puts the tokenized pairs in the buckets depending on size
	data = [[] for _ in _buckets]
	for pair in tokenized_pairs:
		for bucket_id, (source_size,target_size) in enumerate(_buckets):
			if len(pair[0])<source_size and len(pair[1])<target_size:
				data[bucket_id].append([pair[0],pair[1]])
				break
	return data

def pad_pairs(tokenized_pairs,max_len=100):
	# pads all pairs up to a specified length
	print("Padding pairs up to "+str(max_len)+"...")
	x = []
	y = []

	i=0
	for query,response in tokenized_pairs:
		print("Padding pair... "+str(i),end="\r")
		if len(query)==max_len: x.append(query)
		elif len(query)>max_len: x.append(query[:max_len])
		else:
			while len(query)!=max_len:
				query.append(12000)
			x.append(query)

		if len(response)==max_len: y.append(response)
		elif len(response)>max_len: y.append(query[:max_len])
		else:
			while len(response)!=max_len:
				response.append(12000)
			y.append(response)
		i+=1
	print("\nFinished padding pairs.")

	x_list = []
	y_list = []
	for query,response in list(zip(x,y)):
		x_list.append(np.array(query))
		y_list.append(np.array(response))
	
	return x_list,y_list 

def main():
	filename = "data/messages.htm"
	tokenized_pairs,mappings = get_data(filename,verbose=False,load_prior=True)
	x,y = pad_pairs(tokenized_pairs,SEQUENCE_LENGTH)

	input_length = SEQUENCE_LENGTH
	output_length = SEQUENCE_LENGTH
	input_dim = len(x)
	output_dim = len(y)

	if LOAD_PRIOR_MODEL:
		print("Loading model...")
		model = keras.models.load_model("data/model.h5")
	else:
		print("Building model...")
		model = Sequential()
		seqtoseq = SimpleSeq2Seq(
			input_dim = input_dim,
			input_length = input_length,
			output_dim=output_dim,
			output_length=output_length,
			depth=1
			)
		model.add(seqtoseq)
		print("Compiling model...")
		model.compile(loss='mse',optimizer='sgd')
	
	print("Fitting model...")
	model.fit(x,y,batch_size=BATCH_SIZE,nb_epoch=1,show_accuracy=True,verbose=1)
	print("Saving model...")
	model.save("data/model.h5")

	print("\nDone.")

if __name__ == '__main__':
	main()