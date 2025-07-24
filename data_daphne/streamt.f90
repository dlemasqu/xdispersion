PROGRAM streamt 
    IMPLICIT NONE
    INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(12)
    REAL(KIND=dp), ALLOCATABLE :: x(:,:)
    REAL(KIND=dp), ALLOCATABLE :: y(:,:)
    REAL(KIND=dp), ALLOCATABLE :: ux(:,:,:)
    REAL(KIND=dp), ALLOCATABLE :: uy(:,:,:)
    REAL(KIND=dp), ALLOCATABLE :: xp(:,:),yp(:,:)
    REAL(KIND=dp), ALLOCATABLE :: upm(:,:)
    REAL(KIND=dp), ALLOCATABLE :: vpm(:,:)
    REAL(KIND=dp) :: a,b,c,d,e,time,ntemp,ti,tf,diff,mint,pi,up,vp  
    REAL(KIND=dp) :: x1(6),x2(6)  
    REAL(KIND=dp) :: tx1(6),tx2(6)  
    REAL(KIND=dp) :: tu1(6),tu2(6)  
    REAL(KIND=dp) :: t1(6),t2(6)  
    REAL(KIND=dp) :: tabu1(6),tabu2(6),minx,miny  
    REAL(KIND=dp) :: yint,dyint,dt,fmax,dx,dy,xmax,ymax,dumy,umax,amp,rad,th
    INTEGER :: n, nn, i, ii, j, jj, k, m, l, nt, np, nx, ni, nr
    INTEGER :: ip, ip1, jp, jp1 
    CHARACTER*10 :: char

    nx = 100    !piv grid size
    np = 1000  !number of particles
    nt = 200    !number of time steps for trajectory
    ni = 1      !initial time step 
    nr = 1      !number of realisations (starts trajectory at ni=1, then ni+nt, then ni+2*nt etc until ni+nr*nt)

    ALLOCATE(x(nx,nx))
    ALLOCATE(y(nx,nx))
    ALLOCATE(ux(nx,nx,nt))
    ALLOCATE(uy(nx,nx,nt))
    ALLOCATE(xp(np,0:nt))
    ALLOCATE(yp(np,0:nt))
    ALLOCATE(upm(np,2))
    ALLOCATE(vpm(np,2))

    pi = 4.0D0*atan(1.0D0)
    

! Start loop over realisations (nr)    
    do nn=1,nr
     ni=(nn-1)*(nt+1)
    write(*,*) 'ni=',ni 
! Read the data
    write(*,*) ' Reading data ...' 
    OPEN(42,FILE='ExpA_1000.dat',ACTION='READ')
    do n=1,ni
    do i=1,nx
     do j=1,nx
      read(42,*) yint,yint,yint,yint 
     enddo
    enddo
    enddo
    do n=1,nt
    do i=1,nx
     do j=1,nx
      read(42,*) x(i,nx+1-j),y(i,nx+1-j),ux(i,nx+1-j,n),uy(i,nx+1-j,n)
     enddo
    enddo
    enddo
    write(*,*) '              ... OK' 
    close(42)

! (TESTING) Overwrite - Taylor-Green vortex
!    do i=1,nx
!     do j=1,nx
!      ux(i,j,:) = cos(2.0d0*pi*x(i,j))*sin(2.0d0*pi*y(i,j))
!      uy(i,j,:) =-sin(2.0d0*pi*x(i,j))*cos(2.0d0*pi*y(i,j))
!     enddo
!    enddo

    dx = abs(x(2,1)-x(1,1))
    dy = abs(y(1,2)-y(1,1))
    write(*,*) ' dx = ',dx
    write(*,*) ' dy = ',dy
    
! Max coordinates
    xmax = 0.0d0
    ymax = 0.0d0
    do i=1,nx
     do j=1,nx
      if (abs(x(i,j)).gt.xmax) then
              xmax=abs(x(i,j))
      endif
      if (abs(y(i,j)).gt.ymax) then
              ymax=abs(y(i,j))
      endif      
     enddo
    enddo 
    write(*,*) ' xmax = ',xmax       
    write(*,*) ' ymax = ',ymax
    
! Write amplitude and vect 
!    do n=1,nt
!    do i=1,nx,1
!     do j=1,nx,1
!      write(1000+n,*) x(i,j),y(i,j),ux(i,j,n),uy(i,j,n)
!     enddo
!    enddo
!    close(1000+n)
!    enddo

! Max velocity
    umax = 0.0d0
    do i=1,nx
     do j=1,nx
      amp=sqrt(ux(i,j,1)**2.0d0+uy(i,j,1)**2.0d0)
      if (amp.gt.umax) then
              umax=amp
      endif
     enddo
    enddo
    write(*,*) ' umax = ',umax

! Time step
    dt = 0.5d0*dx/umax
    write(*,*) ' dt = ',dt
    dt = 0.05d0

! Initialize particule positions
    call SRAND(1)
    do i=1,np
     !---option 1: polar: distribute particles on 30 radii
     !rad=0.025d0+floor(dble(i)/dble(np)*30.0d0)/30.0d0*0.35d0
     !th=2.0d0*pi*RAND()
     !xp(i,0)=rad*cos(th)
     !yp(i,0)=rad*sin(th)
     
     !---option 2: cartesian: randomly distribute particles over cartesian grid
     xp(i,0)=(RAND()-0.5d0)*0.47d0/0.5d0
     yp(i,0)=(RAND()-0.5d0)*0.47d0/0.5d0 
    enddo
    
!############# START TRAJECTORIES ##############
    do k=1,nt

     write(*,*) ' Iteration ',k,'/',nt

     do n=1,np

! Find closest point in x-y plane
      do i=1,nx-1
       if ( (xp(n,k-1).ge.x(i,1)).AND.(xp(n,k-1).lt.x(i+1,1)) ) then
               ip = i
       endif
      enddo
      do j=1,nx-1
       if ( (yp(n,k-1).ge.y(1,j)).AND.(yp(n,k-1).lt.y(1,j+1)) ) then
               jp = j
       endif
      enddo
      
!If particle too close to the domain border, set velocity to zero
       if (( abs(x(ip,nx/2)).ge.(xmax-4.0d0*dx) ).OR.( abs(y(nx/2,jp)).ge.(ymax-4.0d0*dy) )) then
        xp(n,k)=xp(n,k-1)
        yp(n,k)=yp(n,k-1)
        upm(n,1)=0.0d0
        vpm(n,1)=0.0d0
!If particle not to close to border      
       else
! Build interpolation grid
          do i=1,6
           ip1 = ip+i-3
           tx1(i)=x(ip1,nx/2)
           jp1 = jp+i-3
           tx2(i)=y(nx/2,jp1)
          enddo

!     write(*,*) ' xp = ',xp(n)
!     write(*,*) (tx1(i),i=1,6)
!     write(*,*) ' yp = ',yp(n)
!     write(*,*) (tx2(i),i=1,6)

! Interpolation in the (x,y) plane
	     do i=1,6
	      ip1 = ip+i-3
	      do j=1,6
	       jp1 = jp+j-3
	       tu1(j) = ux( ip1,jp1,k )
	       tu2(j) = uy( ip1,jp1,k )
	      enddo
	      call polint(tx2,tu1,6,yp(n,k-1),tabu1(i),dumy)
	      call polint(tx2,tu2,6,yp(n,k-1),tabu2(i),dumy)
	!      write(*,*) ' Interp 1 ok' 
	     enddo
	     do j=1,6
	      tu1(j) = tabu1(j)
	      tu2(j) = tabu2(j)
	     enddo
	     call polint(tx1,tu1,6,xp(n,k-1),up,dumy)
	     call polint(tx1,tu2,6,xp(n,k-1),vp,dumy)
	!     write(*,*) ' Interp 2 ok' 

! (TESTING) Overwrite interpolation
!      up = cos(2.0d0*pi*xp(n,k-1))*sin(2.0d0*pi*yp(n,k-1))
!      vp =-sin(2.0d0*pi*xp(n,k-1))*cos(2.0d0*pi*yp(n,k-1))

! Integrate trajectory

	     if (k.eq.1) then
	      xp(n,k)=xp(n,k-1)+dt*up
	      yp(n,k)=yp(n,k-1)+dt*vp
	      upm(n,1)=up
	      vpm(n,1)=vp
	     endif

	     if (k.eq.2) then
	      xp(n,k)=xp(n,k-1)+dt*(1.5d0*up-0.5d0*upm(n,1))
	      yp(n,k)=yp(n,k-1)+dt*(1.5d0*vp-0.5d0*vpm(n,1))
	      upm(n,2)=upm(n,1)
	      vpm(n,2)=vpm(n,1)
	      upm(n,1)=up
	      vpm(n,1)=vp
	     endif

	     if (k.gt.2) then
	      xp(n,k)=xp(n,k-1)+dt*((23.0d0/12.0d0)*up-(16.0d0/12.0d0)*upm(n,1)+(5.0d0/12.0d0)*upm(n,2))
	      yp(n,k)=yp(n,k-1)+dt*((23.0d0/12.0d0)*vp-(16.0d0/12.0d0)*vpm(n,1)+(5.0d0/12.0d0)*vpm(n,2))
	      upm(n,2)=upm(n,1)
	      vpm(n,2)=vpm(n,1)
	      upm(n,1)=up
	      vpm(n,1)=vp
	     endif
       endif !End of "if particle not too close to border"

! Write positions
!     if (mod(k,1).eq.0) then
!      write(100+n,*) xp(n),yp(n)
!      write(2000+k,*) xp(n),yp(n)
!      write(3000+k,*) xp(n),yp(n)
!      write(999,*) xp(n),yp(n)
!     endif

! end loop particles
     enddo

! end loop time
    enddo

! Write trajectories
    do n=1,np
    do k=0,nt
     write(1000+nn,*) xp(n,k),yp(n,k)
    enddo
    enddo
    close(1000+nn)

! Write instantaneous positions
!    do k=0,nt
!     do n=1,np
!      write(2000+k,*) xp(n,k),yp(n,k)
!     enddo
!     close(2000+k) 
!    enddo
! Write trace
!    do k=0,nt
!     do n=1,np
!      do j=0,39
!       if ((k-j).ge.0) then
!               write(3000+k,*) xp(n,k-j),yp(n,k-j)
!       endif
!      enddo
!      write(3000+k,*) ' '
!     enddo
!     close(3000+k)
!    enddo

    write(*,*) ' FINITO !' 

! end loop realisations
    enddo    

END PROGRAM streamt 

SUBROUTINE polint(xa,ya,n,x,y,dy)
INTEGER n,NMAX
REAL*8 dy,x,y,xa(n),ya(n)
PARAMETER (NMAX=10)
INTEGER i,m,ns
REAL*8 den,dif,dift,ho,hp,w,c(NMAX),d(NMAX)
ns=1
dif=abs(x-xa(1))
do i=1,n
 dift=abs(x-xa(i))
 if (dift.lt.dif) then
  ns=i
  dif=dift
 endif
 c(i)=ya(i)
 d(i)=ya(i)
enddo
y=ya(ns)
ns=ns-1
do m=1,n-1
 do i=1,n-m
  ho=xa(i)-x
  hp=xa(i+m)-x
  w=c(i+1)-d(i)
  den=ho-hp
  if(den.eq.0.) STOP 'failure in polint' 
  den=w/den
  d(i)=hp*den
  c(i)=ho*den
 enddo
 if (2*ns.lt.n-m)then
  dy=c(ns+1)
 else
  dy=d(ns)
  ns=ns-1
 endif
 y=y+dy
enddo
return
END

! 
