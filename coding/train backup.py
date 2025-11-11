# Training loop
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for i, (sketch, photo) in enumerate(loop):
        sketch, photo = sketch.to(DEVICE), photo.to(DEVICE)
    
        # Ground truths
        real_label = torch.ones_like(real_pred, device=DEVICE)
        fake_label = torch.zeros_like(fake_pred, device=DEVICE)
    
        # ----- Train Discriminator -----
        fake_photo = gen(sketch).detach()
        real_pred = disc(sketch, photo)
        fake_pred = disc(sketch, fake_photo)
        d_loss_real = criterion_GAN(real_pred, real_label)
        d_loss_fake = criterion_GAN(fake_pred, fake_label)
        d_loss = (d_loss_real + d_loss_fake) / 2
        opt_disc.zero_grad()
        d_loss.backward()
        opt_disc.step()
    
        # ----- Train Generator -----
        fake_photo = gen(sketch)
        disc_pred = disc(sketch, fake_photo)
        g_adv = criterion_GAN(disc_pred, real_label)
        g_l1 = criterion_L1(fake_photo, photo)
        g_loss = g_adv + 100 * g_l1  # L1 loss weighted as in Pix2Pix
        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()
    
        loop.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item())
     
    # Save sample outputs and model every few epochs
    if (epoch + 1) % 10 == 0:
        save_image(fake_photo * 0.5 + 0.5, f"{SAVE_DIR}/fake_{epoch+1}.png")
        save_image(photo * 0.5 + 0.5, f"{SAVE_DIR}/real_{epoch+1}.png")
        save_image(sketch * 0.5 + 0.5, f"{SAVE_DIR}/sketch_{epoch+1}.png")
        torch.save(gen.state_dict(), f"{SAVE_DIR}/generator_epoch{epoch+1}.pth")
        torch.save(disc.state_dict(), f"{SAVE_DIR}/discriminator_epoch{epoch+1}.pth")
                                    
                         
print(" Training complete. Model saved.")    
