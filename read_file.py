class read_file:
    def parser(self, line):
        return line[:-1]
    def read(self, file):
        fin = open(file, 'r', encoding="UTF-8")
        while 1:
            line = fin.readline()
            if not line:
                break
            yield self.parser(line)
        fin.close()
        return

class read_file_format_Chi_Squared_data(read_file):
    def parser(self, line):
        return (line[:-1].split('\t')[0], int(line[:-1].split('\t')[1]), int(line[:-1].split('\t')[2]),
                int(line[:-1].split('\t')[3]), int(line[:-1].split('\t')[4]))
class read_file_format_pw_details_semantics(read_file):
    def parser(self, line):
        return line[:-1].split('\t')[0], eval(line[:-1].split('\t')[1]), eval(line[:-1].split('\t')[2])