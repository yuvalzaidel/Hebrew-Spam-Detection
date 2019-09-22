import json
import copy
from emoji import UNICODE_EMOJI
import re
import random
from email.header import decode_header
import urllib.parse

# define letters in a word
chars_english = list(reversed(["A","B","G","D","H","W","Z","X","J","I","K","L","M","N","S","E","P","C","Q","R","F","T","O","U"]))
chars_hebrew = ["ת","ש","ר","ק","צ","פ","ע","ס","נ","מ","ל","כ","י","ט","ח","ז","ו","ה","ד","ג","ב","א"]
final_hebrew_letter = ["ף","ן","ם","ץ","ך"]

names_to_randomize=['גיא','גאי', 'מנח', 'טויטו','טוויטו','טווויטו', 'מנחם', 'יובל', 'זיידל', 'נטע', 'ברק', 'ירדן', 'עמית', 'מתן', 'שי', 'מור', 'אולגה', 'נוהר', 'מאי', 'אלדרי', 'ליאת', 'לסרי', 'אלידור', 'שילה', 'גרשון','טיני','דור','מעוז','נדב','עינת','טל','ליבוביץ','אורית','לוי','נועה','Guy','Twito','טומאר','דולב','סופר','שטרן','הדר',"בוצ'קה",'דןר','אור',"צ'ימבל",'נעם','תומר','זיו','יונתן','ווין','מורן','יואב','פרי','אלמוג','ויין','עודד','אלמוג','רז','קליין','שירלי','יובל','נוהר','דגן','ניר','ליאל','עומר','אורי','שרון','yuval', 'zaidel','Yuval','Zaidel','guy','twito','Guy','Twito']

random_hebrew_names = """אביטוב אביטל אבימלך אבין אבינדב אבינעם אבינר אבינתן אביעד אביעזר אביעם אביקם אביר אבירז אבירם אבירן אבישג אבישור אבישחר אבישי אביתר אבנר
אבשלום אגיל אגם אדיר אדם אדר אהבה אהד אהוב אהובה אהוביה אהוד אהליאב אהרן אודיה אודליה און אופז אופיר אופירה אפק אור אוראל אורפז אור אוראל
אורגד אורה אורטל אורי אוריאל אוריאן אוריאנה אוריה אוריון אורין אורלי אורליה ארן אורנה אורנית אשר אשרה אשרי אושרית אשרת אחוה אחוה אחיאל
אחיאסף אחיה אחיה אחיהו אחיהוד אחיטוב אחינדב אחינעם אחינעם אחיעזר אחיעם אחיקם איה איל אילה איילת איל אילה אילון אלי אילן אילנה אילנית אילת
איריס אירית איתי איתיאל איתמר איתמר איתן איתן אלאור אלדד אלדור אלדר אלה אלה אלול אלמה אלון אלון אלונה אלחנן אלי אליאב אליאור אליאל אליאסף
אליה אליה אליהו אליהוא אלימלך אלינוי אלינור אליסף אליעד אליעזר אליענה אליפז אליצור אליקים אלירם אלירן אלישבע אלישי אלישיב אלישמע אלישע אלמג
אלנתן אלעד אלעזר אלקנה אלרואי אלרום אמונה אמוץ אמיר אמיר אמירה אמתי אמנון אמציה אמרי אמתי אמתי אנאל אנוש אסא אסיף אסנת אסף אפיק אפרים
אציל אצילה אצל אראל אראלה ארבל ארגמן ארד ארז ארי אריאל אריאל אריה אריה אריק ארן ארנה ארנון ארנית אשחר אשירה אשכל אשל אשר אשרה אשרית
אשרת אתגר אתי אסתר באר בארי בינה בלדד בלה בלהה בלומה בן עמי בן בני בניאל בניה בניה בניהו בנימין בעז בצלאל בר ברוך ברך ברוריה ברכה ברעם
ברק בתיה גאלה גאולית גאות גאליה גביר גביש גבריאל גבריאלה גברעם גד גדי גדיאל גדליה גדעון גואל גולדה גולן גיא גיל גלה גל גלבע גלי גליל
גלית גלעד גפן גרשום גרשון גתית דב דבורה דבורית דביר דבירה דבש דגן דגניה דגנית דובב דוד דלב דור דורה דורון דוריאל דותן דין דינה דיצה דליה דן דנה דניאל דניאלה
דפנה דקל דקלה דר דרור דריה דתיה הלל הגות הגר הגרה הדס הדסה הדר הדרה הוד הודיה הושעיה הילי הלה הללי הראל הרצל ורד ורדה ורדינה ורדית זאב זבולון
זהבה זהבית זוהר זוהרה זיו זיוה זיוית זכריה זמורה חביב חביבה חבצלת חגי חגית חגלה חדוה חובב חוה חורב חזקיה חיה חיים חמוטל חנה חנוך חנן חנניה
טהר טובה טוביה טוהר טופז טל טלי טליה טלמור טמיר טמירה יגל יאיר יבין יבניאל יגאל יגיל ידיד ידידה ידידיה יהב יהודה יהודית יהוידע יהונדב יהונתן יהושע יהושפט 
הל יהלום יהלי יואב יואל יובל יובל יוגב יודפת יוחאי יוחנן יוכבד יונה יונת יונתן יוסף יורם יותם יזהר יחזקאל יחיאל ים ימימה ימית ינאי ינון יניר יסכה יסמין יעל יעלה
יעקב יערה יפה יפית יפעה יפעת יפתח יצחק יקותיאל יקיר ירדן ירוחם ירון ירין יריב ירמיה ישורון ישי ישראל כוכב כוכבה כלנית כנרת כפיר כרם כרמי כרמית 
רמל כרמלה כתר לבנה לי לאה לאור לאל לב לבונה לבי לביא לביאה להב לוי ליאון ליאור ליאורה ליאל ליבנה לידור ליה ליהי ליהיא ליטל לילך לימור לינור לירון
לירז לירי מלכה מאור מאי מאיה מאיר מאירה מדן מוטי מור מוריה מזל מזר מטר מיה מיטב מיטל מיכאל מיכה מיכל מילכה מנחם מנשה מעין מרב מרגלית מרדכי
מרים משי מתן מתתיהו משה נאה נאוה נאור נאורה נאות נאמן נבו נבון נגה נדב נהוראי נוגה נוה נווה נוי נועה נח נחום נחמיה נחמן נטע נילי ניסן ניצן ניר ניתאי
נמרוד נעם נעמה נעמי נפתלי נריה נתן נתנאל סגולה סיגל סיגלית סיון סימונה סלעית סמדר סנונית ספיר סתיו עמרי עבדת עברי עבריה עדו עדי עדיאל עדיאלה
עדין עדינה עדן עודד עוז עומר עופר עטרה עידו עידית עילי עלמה עמוס עמיר עמית ענבל ענבר ענת עפרה ערן פורת פז פינחס פלג פנינה צאלה צבי צביה צופיה
צופית קורן קרן קרני ראם ראשית רביד רבקה רואי רום רומי רוני רועי רותם רז רחל רינה רינת רן רנה רננה רפאל שאול שאנן שגיא שהם שובל שוהם שחף שחר
שי שילי שילת שיר שירה שלי שלמה שמשון שני שקד שרה תדהר תהל תום תכלת תמיר תמר""".split()

# my class for treeNodes
class treeNode:

    # set/get node value
    def val(self, val = None):
        if val is None:
            try:
                return self.value
            except:
                return None
        else:
            self.value = val

    # get node childs or add one
    def child(self, chld = None):
        try:
            self.child_list
        except:
            self.child_list = []

        if chld is None:
            return self.child_list
        else:
            self.child_list += [chld]

    # set node child array
    def set_child(self, childs):
        self.child_list = childs

    # remove all childs
    def remove_childs(self):
        self.child_list = []

# print a tree with indent
def graphical_tree(tree, depth = 0):
    print (" ".join(["-"] * depth + [tree.val()]))

    for child in reversed(tree.child()):
        graphical_tree(child, depth+1)

def clean_doubled_white_space(txt):
    return txt
    return re.sub(' +', ' ', txt)

# check if any hebrew in string
def is_hebrew(str):
    if type(str) == type([]):
        if len(str) == 0:
            return False
        else:
            str = str[0]
    global chars_hebrew
    global final_hebrew_letter

    if type(str) != type(""):
        return False

    contains_hebrew = False
    for letter in str:
        if letter in chars_hebrew or letter in final_hebrew_letter:
            contains_hebrew = True

    return contains_hebrew

# function for writing objects to file (for saving train information and save time while checking)
def write_dict_to_file(index,my_dict):
    name = "my_trained" + str(index)
    with open('data/'+name+'.solution', 'w') as trained:
         trained.write(json.dumps(my_dict))

# function for reading objects from file (for reading train information and save time while checking)
def read_dict_from_file(index):
    name = "my_trained" + str(index)
    with open('data/'+name+'.solution', 'r') as trained:
        for data in trained:
            return json.loads(data)

def is_emoji(s):
    return s in UNICODE_EMOJI

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def get_signs(word):
    signs=""
    for c in word:
        s = ord(c)
        if (s >=33 and s<=47) or (s >=58 and s <=64) or (s >= 91 and s <= 96) or (s >= 123 and s<= 126):
            signs += c

    return signs

# determine what string should replace every var type
def replace_var(var, value):
    global emojies_count
    global random_hebrew_names

    end_var = "ENDVAR "

    if var == "mailsubject":
        return " MAILSUBJECT%s "%end_var
    if var == "enter":
        return " ENTERNEWLINE%s "%end_var
    if var == "link":
        return " LINK%d%s "%(len(value),end_var)
    if var == "EMAIL":
        return " EMAIL%s "%end_var
    if var == "emoji":
        return " EMOJI%s%s" % (value, end_var)
    if var == "phone":
        first_number = re.search("[0-9|\\*||\\+|@]", value).start()
        last_number = len(value) - re.search("[0-9|\\*|\\+|@]", value[::-1]).start()

        number_of_digits = sum(c.isdigit() for c in value)
        phoneNumberType = "0"+"(%s)"%value # phone type not found
        if value[0] == "1" and value[2:4] == "00":
            phoneNumberType = "1" # 1800- .... 1700-....
        elif number_of_digits == 4:
            phoneNumberType = "2"  # *1234
        elif number_of_digits == 9:
            phoneNumberType = "3" # 03- ....
        elif number_of_digits == 10:
            phoneNumberType = "4"  # 052-.... 050-....
        elif number_of_digits == 12:
            if "+" in value:
                phoneNumberType = "5"  # +972...
            if "@" in value:
                phoneNumberType = "@"  # @972... (whatsapp tag)

        return "%s PHONENUMBER%s%s %s"%(value[:first_number],phoneNumberType, end_var, value[last_number+1:])
    if var == "name":
        return random.choice(random_hebrew_names)

    return var

# find all vars to be replaced
def replace_forbidden(msg):
    global names_to_randomize

    msg = clean_doubled_white_space(msg).strip()

    link_start_words = ["http:", "https:"]
    link_middle_words = ["www.", "goo.gl", "bit.ly", "youtu.be", "nl.me", "vp4.me", "hyperurl.co","wa.me", ".net"]
    link_words = [".co.il", ".com"]

    links_exist = True
    while links_exist:
        for word in msg.split():
            analyzed_word = word
            if not re.search(".+@.+.com",analyzed_word) and any(ext in analyzed_word for ext in link_start_words+link_middle_words+link_words):
                links_exist = True
                found_start_word = False
                for start_word in link_start_words:
                    if(start_word in analyzed_word):
                        found_start_word = True
                        analyzed_word = start_word + analyzed_word.split(start_word)[1]

                for middle_word in link_middle_words:
                    if(not found_start_word and middle_word in analyzed_word):
                        analyzed_word = analyzed_word.split(middle_word)[0] + middle_word + analyzed_word.split(middle_word)[1]

                msg=msg.replace(analyzed_word, replace_var("link", analyzed_word))
            else:
                links_exist = False

    emails = {}
    for word in msg.split():
        phone_number_regex = [re.search("[^0-9]*1[1-9]00-?[0-9]{3}-?[0-9]{3}[^0-9]*", word)]
        phone_number_regex += [re.search("[^0-9]*0[2-9]-?[0-9]{7}[^0-9]*", word)]
        phone_number_regex += [re.search("[^0-9]*0[2-9][0-9]-?[0-9]{7}[^0-9]*", word)]
        phone_number_regex += [re.search("\\*[0-9]{4}", word)]
        phone_number_regex += [re.search("[0-9]{4}\\*", word)]
        phone_number_regex += [re.search("[\\+|@]972[0-9]{9}", word)]

        number_of_digits = sum(c.isdigit() for c in word)
        if number_of_digits in [4,9,10,12]:
            for regex in phone_number_regex:
                if regex is not None:
                    msg = msg.replace(word, replace_var("phone", word))

        no_ascii_word = ""
        for s in word:
            if not is_ascii(s):
                no_ascii_word += s
        if any(ext == no_ascii_word for ext in names_to_randomize):
            msg = msg.replace(no_ascii_word, replace_var("name", no_ascii_word))
        elif any((no_ascii_word[0:1]=='ו' and ext == no_ascii_word[1:]) for ext in names_to_randomize):
            msg = msg.replace(no_ascii_word, "ו"+replace_var("name", no_ascii_word))


        p = re.compile(r"[^@]+@[^@]+\.[a-zA-z]+")
        for m in p.finditer(word):
            emails[m.group()] = 1

    for email in emails:
        msg = msg.replace(email, replace_var("EMAIL", email))

    msg_temp = msg
    emojies_seen = []
    for s in msg_temp:
        if is_emoji(s):
            if s not in emojies_seen:
                emojies_seen.append(s)
                msg=msg.replace(s,replace_var("emoji", s))



    return msg

# extract subject from mbox datatype
def getSubject(message):
    subject = decode_header(message['subject'])[0][0].decode('utf-8')
    return urllib.parse.unquote(subject)

# extract body from mbox datatype
def getBody(message):  # getting plain text 'email body'
    body = None
    if message.is_multipart():
        for part in message.walk():
            if part.is_multipart():
                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        body = subpart.get_payload(decode=True)
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
    elif message.get_content_type() == 'text/plain':
        body = message.get_payload(decode=True)

    if body is not None:
        body = body.decode('utf-8')
        body = urllib.parse.unquote(body)
        body = body.replace("\r\n\r\n","\r\n")

    return body


# return all the hebrew in string
def hebrew_only(str):
    new_str = ""
    for c in str:
        if (c == " " and len(new_str)>0 and new_str[-1]!=" ") or c in chars_hebrew or c in final_hebrew_letter:
            new_str += c

    return new_str

# average of list without zeros
def average(lst):
    if(len(lst) == 0):
        return 0

    lst = [x for x in lst if x > 0]

    if(len(lst) == 0):
        return 0

    return sum(lst) / len(lst)


def sort_dict(dct):
    return list(reversed(sorted(dct.items(), key=lambda kv: kv[1])))

# pretty print for confusion matrix
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()