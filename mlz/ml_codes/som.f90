subroutine map(XD,ndim,niter,distlib,np,w,importance,npix,aps,ape)
implicit none
integer :: i1,i,it,niter,np,npix,j,k,ndim,best
integer :: ind_rn(0:np-1)
double precision :: distlib(npix,npix),XD(0:np-1,ndim),inputs(ndim)
double precision :: w(ndim,npix),importance(ndim)
double precision :: sigma0,sigma_single,aps,ape
double precision :: par,Hn,bigN,tt,alpha,sig,Htemp

!f2py intent(in) niter,np,npix,distlib,w,XD,importance,ndim,aps,ape
!f2py intent(in,out,copy) w
!f2py depend(npix) distlib,w
!f2py depend(ndim) XD,w,importance
!f2py depend(np) XD


bigN=1.0d0*niter*np

do i=0,np-1
 ind_rn(i)=i
enddo

tt=0.0d0
sigma_single=minval(distlib,mask=distlib .gt. 0)
sigma0=maxval(distlib)

do it=1,niter
 call shuffle(ind_rn,np)
 alpha=par(tt,aps,ape,bigN)
 sig=par(tt,sigma0,sigma_single,bigN)
 do j=0,np-1
  tt=tt+1.0d0
  k=ind_rn(j)
  inputs=XD(k,:)
  call som_best_cell(inputs,w,npix,ndim,importance,best)
  do i1=1,npix
   Htemp=Hn(best,i1,distlib,sig,npix)
   w(:,i1)=w(:,i1)+alpha*Htemp*(inputs-w(:,i1))
  enddo

 enddo
enddo
end
!!!!!!!________________________________
subroutine map_b(XD,ndim,niter,distlib,np,w,importance,npix)
implicit none
integer :: i1,i,it,niter,np,npix,j,k,ndim,best
double precision :: distlib(npix,npix),XD(0:np-1,ndim),inputs(ndim),importance(ndim)
double precision :: w(ndim,npix),accum_w(ndim,npix),accum_n(npix)
double precision :: sigma0,sigma_single
double precision :: par,Hn,bigN,tt,sig,Htemp

!f2py intent(in) niter,np,npix,distlib,w,XD,ndim
!f2py intent(in,out,copy) w
!f2py depend(npix) distlib,w
!f2py depend(ndim) XD,w,importance
!f2py depend(np) XD
bigN=1.0d0*niter*np
tt=0.0d0
sigma_single=minval(distlib,mask=distlib .gt. 0)
sigma0=maxval(distlib)

do it=1,niter
 sig=par(tt,sigma0,sigma_single,bigN)
 accum_w=0.0d0
 accum_n=0.0d0
 do j=0,np-1
  tt=tt+1.0d0
  inputs=XD(j,:)
  call som_best_cell(inputs,w,npix,ndim,importance,best)
  do i1=1,npix
   Htemp=Hn(best,i1,distlib,sig,npix)
   accum_n(i1)=accum_n(i1)+Htemp
   do k=1,ndim
    accum_w(k,i1)=accum_w(k,i1)+Htemp*inputs(k)
   enddo
  enddo
 enddo
 do k=1,ndim
  w(k,:)=accum_w(k,:)/accum_n(:)
 enddo
enddo
end
!----------------------------------------------------




function par(t,alphas,alphae,bigN)
implicit none
double precision par,alphas,alphae,t,bigN
par=alphas*((alphae/alphas)**(t/bigN))
end

function Hn(bmu,bindex,mapD,sigma,npix)
implicit none
integer :: bmu,bindex,npix
double precision :: Hn,sigma,mapD(npix,npix)
Hn=exp(-1.0d0*(mapD(bmu,bindex)**2)/(sigma**2))
end

subroutine shuffle(ind_rn,np)
implicit none
integer :: i,j,trn,tmp
integer, intent(in):: np
integer, intent(inout)::ind_rn(0:np-1)
real :: rn

do i=0,np-1
 call random_number(rn)
 trn=int(rn*np)
 j=i+mod(trn,(np-i))
 tmp=ind_rn(i)
 ind_rn(i)=ind_rn(j)
 ind_rn(j)=tmp
enddo
end subroutine

subroutine som_best_cell(inp,wei,npix,ndim,importance,best)
implicit none
integer :: i
integer, intent(in) :: npix,ndim
double precision, intent(in):: inp(ndim), wei(ndim,npix),importance(ndim) 
integer, intent(out):: best
double precision :: act,act2

act=10000000.0d0

do i=1,npix
 act2=sum(importance(:)*(inp(:)-wei(:,i))**2)
 if (act2 .lt. act) then
  act=act2
  best=i
 endif
enddo


end subroutine



