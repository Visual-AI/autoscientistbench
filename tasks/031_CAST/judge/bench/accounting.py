"""Frozen independent replay of the authors' next-close accounting."""
import csv
import numpy as np
PANELS=('NASDAQ','CSI300','TPX100','Global30')
COLS=['id']+[f'u{i:02d}' for i in range(30)]
def read_orders(path):
 rows={}
 with open(path,newline='') as f:
  r=csv.DictReader(f)
  if r.fieldnames!=COLS:raise ValueError('Expected id and 30 ordered u columns')
  for row in r:
   if None in row or any(v is None for v in row.values()):raise ValueError('Bad row width')
   key=row['id']
   if key in rows:raise ValueError('Duplicate id')
   arr=np.array([float(row[c]) for c in COLS[1:]])
   if not np.isfinite(arr).all():raise ValueError('Nonfinite orders')
   rows[key]=arr
 return rows

def replay(prices,start,orders):
 T,M=prices.shape
 if orders.shape!=(T-start-1,M):raise ValueError('Order shape')
 nav=np.full((T-start,M),1000.)
 position=np.zeros(M);cash=np.full(M,1000.);pending=np.zeros(M);stopped=np.zeros(M,dtype=bool)
 turnover=[];violations=[]
 for k in range(start,T):
  position+=pending
  cash-=pending*prices[k]
  value=position*prices[k]+cash
  liquidate=(~stopped)&(value<200.)
  cash[liquidate]+=position[liquidate]*prices[k,liquidate]
  position[liquidate]=0.;value[liquidate]=cash[liquidate];stopped|=liquidate
  nav[k-start]=value
  if k<T-1:
   pending=orders[k-start]
   if np.any(np.abs(pending[stopped])>1e-8):violations.append('trades_after_margin_stop')
   if np.any(np.abs((position+pending)*prices[k])>value+1e-7):violations.append('leverage')
   with np.errstate(divide='ignore',invalid='ignore'):turnover.append(np.abs(pending)*prices[k]/value)
 portfolio=nav.mean(axis=1)
 with np.errstate(divide='ignore',invalid='ignore'):
  ret=np.diff(portfolio)/portfolio[:-1]
  sd=ret.std(ddof=1)
  sharpe=float(np.sqrt(252.)*ret.mean()/sd) if sd>0 else float('inf')
  mdd=float(np.max((np.maximum.accumulate(portfolio)-portfolio)/np.maximum.accumulate(portfolio)))
  to=float(np.mean(turnover))
 return dict(sharpe=sharpe,mdd=mdd,turnover=to,insolvent_accounts=int(np.any(nav<=0,axis=0).sum()),margin_stopped=int(stopped.sum()),minimum_nav=float(nav.min()),violations=sorted(set(violations))),nav

def evaluate(pred,labels,enforce=True):
 rows=read_orders(pred);z=np.load(labels,allow_pickle=False)
 if tuple(z['panels'])!=PANELS:raise ValueError('Expected four panels')
 wanted=[i for p in PANELS for i in z[p+'_ids'].tolist()]
 if set(rows)!=set(wanted) or len(rows)!=len(wanted):raise ValueError('ID mismatch')
 details={};failed=[]
 for p in PANELS:
  u=np.stack([rows[i] for i in z[p+'_ids']]);d,_=replay(z[p+'_prices'],int(z[p+'_start']),u)
  d['mdd_limit']=float(z[p+'_baseline_mdd'])+.01
  d['turnover_limit']=2*float(z[p+'_baseline_turnover'])
  if d['insolvent_accounts']:failed.append(p+': insolvency')
  if d['violations']:failed.append(p+': trading rules')
  if enforce and not d['mdd']<=d['mdd_limit']+1e-12:failed.append(p+': drawdown')
  if enforce and not d['turnover']<=d['turnover_limit']+1e-12:failed.append(p+': turnover')
  if not all(np.isfinite(d[k]) for k in ['sharpe','mdd','turnover']):failed.append(p+': nonfinite metric')
  details[p]=d
 mean=float(np.mean([d['sharpe'] for d in details.values()]))
 return dict(sharpe=float('inf') if failed else mean,unguarded_sharpe=mean,guards_ok=not failed,failures=failed,panels=details)
