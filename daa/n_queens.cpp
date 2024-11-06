#include <bits/stdc++.h>

using namespace std;
class Solution {
  public:
    void solve(int col, vector < string > & board, vector < vector < string >> & ans, vector < int > & leftrow, vector < int > & upperDiagonal, vector < int > & lowerDiagonal, int n1,int n,bool b,int fqcol) {
      if (col == n1) {
          if(b)
          {
                 ans.push_back(board);
                 return;
          }
          else
          {
              solve(0, board, ans, leftrow, upperDiagonal, lowerDiagonal, fqcol,n,1,fqcol);
          }
     
        return;
      }
      for (int row = 0; row < n; row++) {
        if (leftrow[row] == 0 && lowerDiagonal[row + col] == 0 && upperDiagonal[n - 1 + col - row] == 0) {
          board[row][col] = 'Q';
          leftrow[row] = 1;
          lowerDiagonal[row + col] = 1;
          upperDiagonal[n - 1 + col - row] = 1;
          solve(col + 1, board, ans, leftrow, upperDiagonal, lowerDiagonal, n1,n,b,fqcol);
          board[row][col] = '.';
          leftrow[row] = 0;
          lowerDiagonal[row + col] = 0;
          upperDiagonal[n - 1 + col - row] = 0;
        }
      }
    }

  public:
    vector < vector < string >> solveNQueens(int n) {
      vector < vector < string >> ans;
      vector < string > board(n);
      string s(n, '.');
      
      for (int i = 0; i < n; i++) {
        board[i] = s;
      }
      
      int x,y;
      cout<<"Enter position of first queen: "<<endl;
      cin>>x>>y;
      board[x][y] = 'Q';
      vector < int > leftrow(n, 0), upperDiagonal(2 * n - 1, 0), lowerDiagonal(2 * n - 1, 0);
      leftrow[x]=1;
      upperDiagonal[n - 1 + y - x]=1;
      lowerDiagonal[x + y] = 1;
      
      
      solve(y+1, board, ans, leftrow, upperDiagonal, lowerDiagonal, n,n,0,y);
      
      
      
      return ans;
    }
};
int main() {
  int n = 8; // we are taking 4*4 grid and 4 queens
  Solution obj;
  vector < vector < string >> ans = obj.solveNQueens(n);
  for (int i = 0; i < ans.size(); i++) {
    cout << "Arrangement " << i + 1 << "\n";
    for (int j = 0; j < ans[0].size(); j++) {
      cout << ans[i][j];
      cout << endl;
    }
    cout << endl;
  }
  return 0;
}