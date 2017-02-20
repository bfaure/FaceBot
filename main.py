import datetime
import sys

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

		if self.is_url==False: self.fix_text()
		self.fix_user()
		self.fix_timestamp()

		self.process_data()

	def fix_timestamp(self):
		# gets seconds since epoch from the timestamp string

		broken = self.timestamp.split(",")

		month = get_int_from_month(broken[1].split(" ")[1])
		day = int(broken[1].split(" ")[2])
		year = int(broken[2].split(" ")[1])
		hour = int(broken[2].split(" ")[3].split(":")[0])
		minute = broken[2].split(" ")[3].split(":")[1]
		if minute.find("pm"): hour+=12
		if hour==24: hour=23
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
		self.user.replace("&#064;","@")

	def fix_text(self):
		# swaps out any known Facebook character codes with their appropriate characters
		char_codes 	= ["&#039;",	"&quot;",	"&lt;",	"&gt;",	"&#123;",	"&#125;"] 
		chars 		= ["\'",		"\"",		"<",	">",	"{",		"}"]
		for code,correct in list(zip(char_codes,chars)):
			self.text = self.text.replace(code,correct)

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
		print(self.horizontal_border)

def read_messages(filename,count=-1,display=False):
	f = open(filename,"r", encoding='utf8')
	text = f.read()
	print("Reading messages...")
	messages = text.split("<div class=\"message\"><div class=\"message_header\"><span class=\"user\">")
	print("Found "+str(len(messages))+" messages in "+filename)
	parsed_messages = []
	if count==-1: count=1000000
	print("Parsing messages...")
	i=0
	messages = messages[1:] # first entry isnt a message

	for message in messages:
		if i>count: break
		print("Parsing messages... "+str(i),end="\r")
		m = message_t(message)
		parsed_messages.append(m)
		i+=1

	print("\nFinished parsing "+str(len(parsed_messages))+" messages.")

	if display:
		num_to_print = 3
		print("Printing the first "+str(num_to_print)+" messages...")
		for j in range(num_to_print):
			print("\n")
			parsed_messages[j].display()
		print("\n")
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
	print("Sorting messages...")
	sorted_messages = sorted(message_ts, key=lambda message: message.time_in_seconds)
	print ("Finished sorting.")
	return sorted_messages

def main():
	filename = "messages.htm"
	dest = "messages.txt"
	messages = read_messages(filename,count=-1,display=True)
	messages = sort_messages(messages)
	save_messages(dest,messages)

if __name__ == '__main__':
	main()