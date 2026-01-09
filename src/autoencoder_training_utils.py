from src.autoencoder import StandardAutoencoder, TimeConsistentAutoencoder, BiLevelTimeAutoencoder

def train_standard_ae(model: StandardAutoencoder, dataloader, criterion, optimizer, device, num_epochs):
    print("\nStarting Standard Autoencoder Training...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, x_i in enumerate(dataloader):
            # x_i is the single frame (Frame i)
            x_i = x_i.to(device)
            optimizer.zero_grad()
            
            # 1. Forward Pass
            z_i, x_prime_i = model(x_i)
            # 2. Calculate Loss
            loss = criterion(x_prime_i, x_i)
            
            # 3. Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n--- Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.6f} ---\n")



def train_time_consistent_autoencoder(model: TimeConsistentAutoencoder, dataloader, criterion, optimizer, device, num_epochs, lambda_recon=1.0, lambda_cons=0.001):
    print("\nStarting Time Consistent Autoencoder Training...")
    
    # 1. Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0

        # dataloader yields (x_i, x_i_plus_1)
        for batch_idx, data in enumerate(dataloader):
            # Move data to device
            x_i = data[:,0].to(device).float()
            x_i_plus_1 = data[:,1].to(device).float()
            optimizer.zero_grad()
            
            # 2. Forward Pass: Computes all five required outputs
            z_t, x_t_prime, z_t_plus_1, x_t_plus_1_prime = model(x_i, x_i_plus_1)
            
            # 3. Calculate the Loss Terms
            
            # A. Reconstruction Loss (L_recon)
            L_Recon = criterion(x_t_prime, x_i) + criterion(x_t_plus_1_prime, x_i_plus_1)
            
            # B. Consistency Loss (L_time)
            L_Cons = criterion(z_t, z_t_plus_1)

            # 4. Total Loss and Optimization
            total_batch_loss = (lambda_recon * L_Recon) + (lambda_cons * L_Cons)
            # print(total_batch_loss)
            total_batch_loss.backward()
            optimizer.step()
            
            epoch_loss += total_batch_loss.item()
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n--- Epoch {epoch+1} Finished. Avg Total Loss: {avg_epoch_loss:.5f} ---\n")



def train_bilevel_time_autoencoder(model: BiLevelTimeAutoencoder, dataloader, criterion, optimizer, device, num_epochs, lambda_recon=1.0, lambda_comm=0.001):
    print("\nStarting BiLevel Time Autoencoder Training...")
    
    # 1. Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0

        # dataloader yields (x_i, x_i_plus_1)
        for batch_idx, data in enumerate(dataloader):
            # Move data to device
            x_i = data[:,0].to(device).float()
            x_i_plus_1 = data[:,1].to(device).float()
            optimizer.zero_grad()
            
            # 2. Forward Pass: Computes all five required outputs
            z_t, x_t_prime, z_t_plus_1, z_t_pred, x_t_plus_1_prime = model(x_i, x_i_plus_1)
            
            # 3. Calculate the Loss Terms
            
            # A. Reconstruction Loss (L_recon)
            L_Recon = criterion(x_t_prime, x_i) + criterion(x_t_plus_1_prime, x_i_plus_1)
            
            # B. Consistency Loss (L_time)
            L_Comm = criterion(z_t_plus_1, z_t_pred)

            # 4. Total Loss and Optimization
            total_batch_loss = (lambda_recon * L_Recon) + (lambda_comm * L_Comm)
            # print(total_batch_loss)
            total_batch_loss.backward()
            optimizer.step()
            
            epoch_loss += total_batch_loss.item()
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n--- Epoch {epoch+1} Finished. Avg Total Loss: {avg_epoch_loss:.5f} ---\n")
