# coding:utf-8
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        if matrix is None or path is None:
            return False
        if matrix == path:
            return True
        m = [matrix[i*cols:(i+1)*cols] for i in range(rows)]
        print(m)
        return self.move(m, 0, 0, path)
    
    def move(self, matrix, i, j, path):
        """
        这个函数用来全局进行移动
        i行数 j列数
        """
        if i >= len(matrix) or j >= len(matrix[0]) or i < 0 or j < 0 :
            return False
        flag_core = [[0 for a in range(len(matrix[0]))] for b in range(len(matrix))]
        return self.move_core(matrix, i, j, 0, path, flag_core) or self.move(matrix, i+1, j, path) or  self.move(matrix, i, j+1, path)
    
    def move_core(self, matrix, i, j, k, path, flag):
        """
        matrix[i][j]与path的开头进行匹配
        """
        #print('move_core: i', i, 'j', j)
        if i >= len(matrix) or j >= len(matrix[0]) or i < 0 or j < 0:
            return False
        if k == len(path):
            return True
        if flag[i][j] == 1:
            return False
        if matrix[i][j] != path[k]:
            return False
        else:
            print('move_core success: i', i, 'j', j, 'k', path[k])
            flag[i][j] = 1
            return self.move_core(matrix, i+1, j, k+1, path, flag) or self.move_core(matrix, i, j+1, k+1, path, flag) or self.move_core(matrix, i-1, j, k+1, path, flag) or self.move_core(matrix, i, j-1, k+1, path, flag)

if __name__ == '__main__':
    s = Solution()
    # "ABCESFCSADEE",3,4,"ABCCED"
    matrix = "ABCESFCSADEE"
    print(s.hasPath("ABCEHJIGSFCSLOPQADEEMNOEADIDEJFMVCEIFGGS",5,8,"SLHECCEIDEJFGGFIE"))